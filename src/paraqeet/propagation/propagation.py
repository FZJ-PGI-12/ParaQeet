"""Abstract base classes for solvers that propagate the equation of motion and its gradient in time."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Any, override

import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import scan

from paraqeet.differentiable import Differentiable
from paraqeet.propagation.utils import construct_batched_times, construct_times
from paraqeet.quantity import Array, Float


class Propagation(ABC):
    """Abstract base class for solvers of the equations of motion.

    The implemented propagation methods use vectorized operations to evaluate evolution of a
    single state, batch of states or entire propagator with the same code.
    For evaluating propagation of a propagator ``initial_state``
    can also be set to an initial `propagator`.

    To implement new propagation methods, users are required to implement the ``_propagate`` method
    that propagates a quantum state/propagator from some initial to final time, under some EOM.
    Default implementation includes a batched propagation method that computes the EOM for the interpolated
    time grid, and performs the entire propagation in a single compiled loop. Utilizing this requires a
    **uniform time grid** to ensure a single dt and to avoid recompilation.

    Note:
        `Propagator` here refers to both unitary matrices for closed system and quantum channel for
        open quantum systems.

    Note:
        By default we use a ``batched_propagation = True`` that computes the EOM for the entire
        interpolated time grid and performs a jitted propagation (for speed). For cases that are
        limited by RAM, set ``batched_propagation = False``.
    """

    _eom_func: Callable[[Array], Array]
    _resolution: float
    _initial_state: Array
    _batched_propagation: bool = True

    def __init__(self, eom_func: Callable[[Array], Array], resolution: float, initial_state: Array) -> None:
        """
        Args:
            eom_func: A function that gives the equation of motion.
            resolution: Propagation resolution used to solve the equation of motion.
                The corresponding time step dt = 1/resolution
            initial_state: State at the beginning of the simulation.
        """
        self._eom_func = eom_func
        self._resolution = resolution
        self.initial_state = initial_state

    @property
    def batched_propagation(self) -> bool:
        """Return whether the EOM is sampled for all times and propagated in one compiled call."""
        return self._batched_propagation

    @batched_propagation.setter
    def batched_propagation(self, batched_propagation: bool) -> None:
        """Set whether the EOM is sampled for all times and propagated in one compiled call.

        Note:
            Setting ``batched_propagation = True`` uses more RAM. For memory sensitive tasks
            set this to ``False``.
        """
        self._batched_propagation = batched_propagation

    @property
    def resolution(self) -> float:
        """Return the propagation resolution."""
        return self._resolution

    @resolution.setter
    def resolution(self, resolution: float) -> None:
        """Set the propagation resolution."""
        self._resolution = resolution

    @property
    def initial_state(self) -> Array:
        """Return initial state."""
        return self._initial_state

    @initial_state.setter
    def initial_state(self, state: Array) -> None:
        """Set initial state.

        The initial states can be thought of a collection of column vectors and
        all of them are propagated simultaneously by multiplying them with the `propagator`.
        So the initial states can in general would be a n x m matrix,
        where n is the system dimension and m are the number of initial states chosen.
        """
        # TODO: Provide explicit wrappers for multiple initial states or density vectors
        self._initial_state = jnp.array(state, dtype=jnp.complex128)

    def _construct_time_grid(self, step_times: Array, dt: Float) -> Array:
        """Return the times at which the EOM has to be sampled for one propagation step.

        Default implementation samples EOM at the midpoint of every step.

        Methods such as ODE solvers, override this and return more than one sample
        per entry of ``step_times``.

        Args:
            step_times: Times at which the propagation steps start.
            dt: Length of one propagation step.

        Returns:
            Times at which ``_eom_func`` is evaluated.
        """
        return step_times + dt / 2

    def _construct_batched_time_grid(self, times: Array) -> tuple[Array, Float, Array] | None:
        """Return the times at which the EOM has to be sampled, for all segments at once.

        Args:
            times: Array of times.

        Returns:
            Batched time grid, dt, a steps array for propagation. Returns None if times is not
            a uniform grid.

        """
        if not self.batched_propagation:
            return None

        batched_times_and_dt = construct_batched_times(times, self._resolution)
        if batched_times_and_dt is None:
            return None
        batched_times, dt = batched_times_and_dt

        batched_time_grid = vmap(self._construct_time_grid, in_axes=(0, None))(batched_times, dt)
        return batched_time_grid, dt, jnp.arange(0, batched_times.shape[1], 1)

    def _propagate_args(self, dt: Float) -> tuple[Array, ...]:
        """Return additional arguments that ``_propagate`` needs, after the ``steps`` argument.

        Args:
            dt: Length of one propagation step.
        """
        return ()

    @abstractmethod
    def _propagate(self, eom: Array, state: Array, steps: Array, *args: Any, **kwargs: Any) -> Array:
        """Propagate the state/propagator in time by solving the EOM.

        Note:
            It is recommended to make this function a 'pure' JAX function
            supporting JIT and gradient using vjp (reverse-mode AD).
            The type of arguments should be jax.Array.
            It is recommended to the user to add a `@jit` decorator with the
            appropriate `static_argnums`.
            For gradients using AD, derivative of this method is computed with
            respect to the first parameter (eom).

        Args:
            eom: Equation of motion as an Array, sampled at ``_construct_time_grid``.
            state: Initial state/propagator for propagation
            steps: Array of indices to iterate over.
            *args: Extra arguments as returned by ``_propagate_args``.
            **kwargs: Unused, kept for subclasses that might need keyword arguments.

        Returns:
            Propagated state/propagator by solving the EOM.
        """
        pass

    def _sample_eom_batched(self, times: Array) -> tuple[Array, Float, Array, Array] | None:
        """Sample the equation of motion of all segments in a single call.

        Evaluating the EOM costs a fixed overhead per call, which dominates the runtime when paid
        once per segment.

        Args:
            times: Array of times.

        Returns:
            The EOM with the segment along the first and the sample along the second axis, already
            scaled with the step size; dt; the iteration indices of one segment; and
            the times the EOM was sampled at.
        """
        grid = self._construct_batched_time_grid(times)
        if grid is None:
            return None
        batched_time_grid, dt, steps = grid

        eom = self._eom_func(jnp.reshape(batched_time_grid, (-1,))) * dt
        eom_batched = jnp.reshape(eom, batched_time_grid.shape + eom.shape[1:])

        return eom_batched, dt, steps, batched_time_grid

    @partial(jit, static_argnums=(0,))
    def _propagate_batched(self, eom: Array, state: Array, steps: Array, *args: Any) -> Array:
        """Propagate a state through all segments in a single compiled loop.

        Args:
            eom: Equation of motion of every segment, as returned by ``_sample_eom_batched``.
            state: State/propagator at the beginning of the first segment.
            steps: Array of indices to iterate over within one segment.
            *args: Extra arguments as returned by ``_propagate_args``.

        Returns:
            The state at every time point, the given initial one included, with time along the
            first axis.
        """

        def propagate_batched(state: Array, eom_batched: Any) -> tuple[Array, Array]:
            state = self._propagate(eom_batched, state, steps, *args)
            return state, state

        _, states = scan(propagate_batched, state, eom)
        propagated_states: Array = jnp.concatenate([jnp.expand_dims(state, axis=0), states])
        return propagated_states

    def get_value(self, times: Array) -> Array:
        """Return the solution of the equations of motion.

        Loop over all desired times in time at set resolution.

        The first dimension of the result will always be the time.
        Like in the model, the format of the other dimensions depends on the
        implementation and could for example be a propagated state vector or
        a propagator in matrix form.

        If ``batched_propagation`` is ``True``, ``_propagate_batched`` method is used
        to compute all the EOM at once, and propagate in one compiled loop.
        *Input ``times`` has to be uniformly spaced in this case.*

        Args:
            times: Array of times.

        Returns:
            Array: Returns the solution of the equations of motion.

        Raises:
            ValueError: If fewer than two time points are given.

        """
        if len(times) < 2:
            raise ValueError("Propagation.get_value needs at least two time points.")

        sampled_eom = self._sample_eom_batched(times)
        if sampled_eom is not None:
            eom, dt, steps, _ = sampled_eom
            propagated_states: Array = self._propagate_batched(
                eom, self._initial_state, steps, *self._propagate_args(dt)
            )
            return propagated_states

        # Fallback to python loop for propagation: Uses less RAM.

        psis = [self._initial_state]

        for ti in range(1, len(times)):
            step_times, dt = construct_times(times, ti, self._resolution)
            eom = self._eom_func(self._construct_time_grid(step_times, dt)) * dt
            psis_t = self._propagate(
                eom,
                psis[ti - 1],
                jnp.arange(0, len(step_times), 1),
                *self._propagate_args(dt),
            )
            psis.append(psis_t)

        psis_arr = jnp.array(psis)
        return psis_arr


class DifferentiablePropagation(Differentiable):
    """Wraps a Propagation object to add gradient computation."""

    _prop: Propagation
    _eom_gradient_func: Callable[[Array], Array]

    def __init__(
        self,
        propagation: Propagation,
        eom_gradient_func: Callable[[Array], Array],
    ):
        """
        Args:
            propagation: Any propagation object.
            eom_gradient_func: Function that returns the gradient of the EOM.
        """
        self._prop = propagation
        self._eom_gradient_func = eom_gradient_func

    def _eom_and_gradient_func(self, times: Array) -> tuple[Array, Array]:
        """Return EOM and gradient of EOM for the given times array.

        Note:
            Default implementation computes the EOM and its gradient separately.
            *For cases where both the EOM and its gradient can be computed simultaneoulsy*
            *overwrite this method for decreasing the computational overhead.*

        Args:
            times: Array of time steps.
        """
        return self._prop._eom_func(times), self._eom_gradient_func(times)

    def _sample_eom_and_gradient_batched(self, times: Array) -> tuple[Array, Array, Float, Array] | None:
        """Sample the EOM and its gradient of all segments in a single call.

        The counterpart of ``Propagation._sample_eom_batched`` for the gradient wrappers,
        which need the derivative of the EOM at the same times.

        Args:
            times: Array of times.

        Returns:
            The EOM and its gradient, both with the segment along the first and the sample along
            the second axis and both scaled with the step size; dt; and the iteration
            indices of one segment.
        """
        grid = self._prop._construct_batched_time_grid(times)
        if grid is None:
            return None
        batched_time_grid, dt, steps = grid

        eom, eom_gradient = self._eom_and_gradient_func(jnp.reshape(batched_time_grid, (-1,)))
        eom = jnp.array(eom) * dt
        eom_gradient = jnp.array(eom_gradient) * dt

        return (
            jnp.reshape(eom, batched_time_grid.shape + eom.shape[1:]),
            jnp.reshape(eom_gradient, batched_time_grid.shape + eom_gradient.shape[1:]),
            dt,
            steps,
        )

    @override
    def get_value(self, times: Array) -> Array:
        """Return the solution of the equations of motion from the propagation method.

        Args:
            times: Array of times.

        """
        return self._prop.get_value(times)

    @override
    @abstractmethod
    def get_gradient(self, times: Array) -> Array:
        """Return gradient of propagated state/propagator w.r.t. specified parameters.

        Args:
            times: Array of time steps.
        """
        pass
