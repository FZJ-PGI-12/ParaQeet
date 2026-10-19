"""Signal base class and signal components such as local oscillators, DRAG mixers, and filters."""

from abc import abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Any, override

import jax
import jax.numpy as jnp
from jax import jacfwd, jit, vmap
from jax.scipy.special import erf

from paraqeet.differentiable import Differentiable
from paraqeet.optimizable import Optimizable
from paraqeet.quantity import Array, Float, Quantity

jax.config.update("jax_enable_x64", True)


class Signal(Optimizable, Differentiable):
    """Classical electronics for signal generation."""

    # AC: the following annotation does not seem needed / correct.
    _partial_grads_function: Callable | None = None
    _gradient_function: Callable | None = None
    _value_function: Callable | None = None
    _grad_arg_nums: tuple[int, ...] = ()

    def _compute_gradient_function(
        self, signal_function: Callable, argnums: tuple[int, ...], vmap_axes: tuple[int | None, ...]
    ) -> None:
        """Return a compute gradient function from the signal function.

        Args:
            signal_function: A function that generates signals.
            argnums: A tuple of ints containing a variable number of argument numbers.
            vmap_axes: A tuple of ints.

        """
        grads = jacfwd(signal_function, argnums=argnums)
        self._partial_grads_function = grads
        partial_grads = vmap(grads, vmap_axes)
        self._gradient_function = jit(partial_grads)

    @override
    def set_optimizable_parameters(self, params: list[Quantity]) -> None:
        """Set optimizable parameters for optimization.

        Args:
            params: Input list of parameters to be set.

        """
        super().set_optimizable_parameters(params)

        self._grad_arg_nums = ()
        for i, param in enumerate(self.get_parameters()):
            if self._is_optimized(param):
                self._grad_arg_nums += (i,)

        # Recompute gradient function
        params = self.get_parameters()
        num_params = len(params)

        # vmap over time axis only, set everything else to None
        vmap_axes = (None,) * num_params
        vmap_axes += (0,)  # type: ignore

        if len(self._grad_arg_nums) > 0:
            self._compute_gradient_function(
                self._evaluate,
                argnums=self._grad_arg_nums,
                vmap_axes=vmap_axes,
            )
        else:
            self._gradient_function = None

    @abstractmethod
    def _evaluate(self, *args: Any, **kwargs: Any) -> Array:
        """Evaluate the output of the system.

        Note:
            It is recommended to make this function a 'pure' JAX function supporting JIT.
            The arguments are supposed to be arranged as (parameters, t), i.e., time after parameters.
            The type of arguments should be jax.Array.
            The output has to be a scalar for a scalar time input to support AD.
            Use jax.squeeze() to remove extra dimensions.
        """
        pass

    @override
    def get_value(self, times: Array) -> Float | Array:
        params = self.get_parameters()
        param_values = [param.get_value() for param in params]
        t_arr = jnp.array(times, ndmin=1)

        # To ensure that the compilation takes place only once
        if self._value_function is None:
            self._value_function = jit(self._evaluate)

        value = self._value_function(*param_values, t_arr)
        # ignoring mypy due to jit
        return value  # type: ignore

    @override
    def get_gradient(self, times: Array) -> Array:
        """Compute the gradient of the ``_evaluate`` method.

        Note:
            This uses Automatic differentiation as a fallback.
            The ``_evaluate`` method should be a **pure** function (should take the
            optimizable parameters as function arguments and doesn't depend on
            global variables).
            Refer to https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
            for functionally **pure** functions.
            To implement analytical gradients / other methods for gradient
            computation overwrite this method in the inherited class.

        Args:
            times: Array of times.

        Returns:
            The gradient array of the ``_evaluate`` method.

        """
        params = self.get_parameters()
        param_values = [param.get_value() for param in params]
        t_arr = jnp.array(times, ndmin=1)
        grads = jnp.empty((t_arr.shape[0], 0))
        if self._gradient_function is not None:
            grad = self._gradient_function(*param_values, t_arr)
            grads = jnp.stack(grad, axis=1)
            grads = jnp.squeeze(grads, -1)
        return grads

    @override
    def get_value_and_gradient(self, times: Array) -> tuple:
        return self.get_value(times), self.get_gradient(times)

    def get_time_gradient(self, times: Array) -> Array:
        """Compute a signal envelope's time derivative.

        Args:
            times: Array of times.

        Returns:
            An array of the signal's time derivative.

        """
        t_arr = jnp.array(times, ndmin=1)
        env_time_grad_fun = jacfwd(self.get_value, argnums=0)
        env_time_grad = vmap(env_time_grad_fun, in_axes=(0,))(t_arr)
        return jnp.squeeze(env_time_grad)

    def get_time_and_parameter_gradient(self, times: Array) -> Array:
        r"""Compute the double derivative with respect to parameter and time.

        This function computes

        .. math::
            \frac{\partial^2 \Omega}{\partial t \partial \alpha}

        for a pulse :math:`\Omega(t)` and parameter :math:`\alpha`.

        Args:
            times: Array of times.

        Returns:
            An array of the signal's time derivative.

        """
        params = self.get_parameters()
        param_values = [param.get_value() for param in params]
        t_arr = jnp.array(times, ndmin=1)
        if self._partial_grads_function is None:
            return jnp.empty((t_arr.shape[0], 0))

        def compute_gradient_vmap(t):
            f = lambda time: jnp.reshape(jnp.stack(self._partial_grads_function(*param_values, time)), (-1,))
            return jacfwd(f)(t)

        # ignoring mypy due to vmap
        return vmap(compute_gradient_vmap)(t_arr)  # type: ignore


class LocalOscillator(Signal):
    """A local oscillator's carrier signal."""

    _lo_freq: Quantity

    def __init__(self, frequency: Quantity | None = None) -> None:
        """
        Args:
            frequency: Frequency of the local oscillator.
        """
        self._lo_freq = frequency or Quantity(
            value=jnp.array(4.8e9 * 2 * jnp.pi),
            min_value=jnp.array(0.8 * 4.8e9 * 2 * jnp.pi),
            max_value=jnp.array(1.2 * 4.8e9 * 2 * jnp.pi),
            unit="Hz",
            name="lo_freq",
            two_pi=True,
        )

    @override
    def get_parameters(self) -> list[Quantity]:
        return [self._lo_freq]

    @property
    def frequency(self) -> Quantity:
        """Return the frequency of the constant oscillating tone."""
        return self._lo_freq

    @frequency.setter
    def frequency(self, frequency: Quantity) -> None:
        """Set the frequency of the constant oscillating tone.

        Args:
            frequency: The frequency of the local oscillator.

        """
        self._lo_freq = frequency

    @partial(jax.jit, static_argnums=(0,))
    def _evaluate(self, freq: Array, times: Array) -> Array:  # type: ignore
        """Calculate the unscaled carrier signal.

        Args:
            freq: The frequency of the carrier signal.
            times: One-dimensional vector of timestamps.

        Returns:
            The unscaled carrier signal.
        """
        return jnp.exp(1j * freq * times)

    @override
    def get_value(self, times: Array) -> Array:
        # returns JitWrapped
        return self._evaluate(self._lo_freq.get_value(), times)  # type: ignore

    @override
    def get_gradient(self, times: Array) -> Array:
        freq = self._lo_freq.get_value()
        t_arr = jnp.array(times, ndmin=1)

        grads = jnp.empty((t_arr.shape[0], 0))
        if self._is_optimized(self._lo_freq):
            grads = jnp.reshape(1j * t_arr * self._evaluate(freq, t_arr), (-1, 1))

        return grads

    def get_time_gradient(self, times: Array) -> Array:
        """Compute a signal's time derivative.

        Args:
            times: Array of times.

        Returns:
            An array of the signal's time derivative.

        """
        freq = self._lo_freq.get_value()
        # returns JitWrapped
        return 1j * times * self._evaluate(freq, times)  # type: ignore


class DRAGMixer(Signal):
    """A DRAG correction :cite:p:`motzoi2009simple` applied to signal envelope.

    The DRAG component is calculated for a set of envelopes and added in the
    orthogonal direction in the x-y plane.

    """

    def __init__(
        self,
        envelopes: Signal | list[Signal],
        deltas: list[Quantity] | None = None,
    ) -> None:
        """
        Args:
            envelopes: The list of shape defining signal envelopes.
            deltas: The delta parameter by which to shift the frequency of the DRAG
                component.
        """
        self._envs = envelopes if isinstance(envelopes, list) else [envelopes]
        DRAGMixer._add_deltas(self._envs, deltas)

    @override
    def get_parameters(self) -> list[Quantity]:
        params = list()
        for tone in self._envs:
            params += tone.get_parameters()
            params += [DRAGMixer._get_tone_delta(tone)]
        return params

    def get_envelopes(self) -> list[Signal]:
        """Return envelopes from the DRAGMixer."""
        return self._envs

    @staticmethod
    def _add_deltas(envelope_tones: list[Signal], deltas: list[Quantity] | None) -> None:
        """Add a DRAG delta parameter Quantity to each envelope Tone.

        Args:
            envelope_tones: The list of tones defining the total envelope.
            deltas: A List of Quantities representing the delta parameters to add to
                each envelope Tone.

        Returns:
            The list of envelope Tones with the added delta parameters.
        """
        for ii, env_tone in enumerate(envelope_tones):
            env_tone.__setattr__(
                "_delta",
                deltas[ii]
                if deltas
                else Quantity(
                    jnp.array(-200e6 * 2 * jnp.pi),
                    min_value=jnp.array(-3 * 200e6 * 2 * jnp.pi),
                    max_value=jnp.array(-0.1 * 200e6 * 2 * jnp.pi),
                    unit="Hz",
                    name="Delta",
                ),
            )

    @staticmethod
    def _get_tone_delta(tone: Signal) -> Any:
        """Return a list of deltas for each tone.

        Args:
            tone: The signal envelope.

        Returns:
            List of delta values for each tone.
        """
        return tone.__getattribute__("_delta")

    def _evaluate(self, times: Array, *deltas: Array) -> Array:
        """Compute the DRAG Envelope using deltas.

        Explicit function depending on deltas to compute gradients using AD.

        Args:
            times: Array of times.
            deltas: Variable number of inputs for delta parameters for each tone.

        Returns:
            An array of signals of the DRAG envelope.
        """
        total_env = jnp.zeros_like(times, dtype=jnp.complex128)
        for delta, tone in zip(deltas, self._envs):
            env = tone.get_value(times)
            env_grad = tone.get_time_gradient(times)
            total_env += env - 1.0j / delta * env_grad
        return jnp.squeeze(total_env)

    @override
    def get_value(self, times: Array) -> Array:
        deltas = [DRAGMixer._get_tone_delta(tone).get_value() for tone in self._envs]
        return self._evaluate(times, *deltas)

    def set_optimizable_parameters(self, params: list[Quantity]) -> None:
        """Set specified parameters to be optimized.

        Also add the indices to ``_grad_arg_nums`` to compute the gradients.

        Args:
            params: Input list of parameters to be set.
        """
        super().set_optimizable_parameters(params)

        for tone in self._envs:
            tone.set_optimizable_parameters(params)

    @override
    def get_gradient(self, times: Array) -> Array:
        deltas = [DRAGMixer._get_tone_delta(tone) for tone in self._envs]
        delta_values = [delta.get_value() for delta in deltas]
        times_arr = jnp.array(times, ndmin=1)

        gradients = jnp.zeros(shape=(times_arr.shape[0], 0))

        # Collect gradients wrt envelope parameters
        for i, tone in enumerate(self._envs):
            _, grads = tone.get_value_and_gradient(times_arr)
            mixed_der = tone.get_time_and_parameter_gradient(times_arr)
            grads += -1.0j / delta_values[i] * mixed_der
            gradients = jnp.append(gradients, grads, axis=1)

        # Collect gradients wrt deltas
        for i, tone in enumerate(self._envs):
            if self._is_optimized(deltas[i]):
                grad = 1j / (delta_values[i] ** 2) * tone.get_time_gradient(times_arr)
                grad = jnp.expand_dims(grad, axis=1)
                gradients = jnp.append(gradients, grad, axis=1)

        return jnp.array(gradients)


class FlatTopGaussianFilter(Signal):
    """A shape filter that forces the pulse to smoothly start and end at zero.
    This filter multiplies the input pulse with a flat-top Gaussian pulse.

    Note:
        Use filters before the generators.
        Otherwise, automatic differentiation does not work in the current setup.

    This is similar to ``PWCGenerator.multiply_flat_top = True``.
    """

    _envs: list[Signal]
    _t_final: Quantity

    def __init__(self, envelopes: Signal | list[Signal], t_final: Quantity) -> None:
        """
        Args:
            envelopes: The list of shape defining signal envelopes.
            t_final: Final time.
        """
        self._envs = envelopes if isinstance(envelopes, list) else [envelopes]
        self._t_final = t_final

    @override
    def get_parameters(self) -> list[Quantity]:
        params = list()
        for tone in self._envs:
            params += tone.get_parameters()
        return params

    def get_envelopes(self) -> list[Signal]:
        """Return envelopes from the FlatTopGaussianFilter."""
        return self._envs

    def set_optimizable_parameters(self, params: list[Quantity]) -> None:
        """Set specified parameters to be optimized.

        Also add the indices to ``_grad_arg_nums`` to compute the gradients.

        Args:
            params: Input list of parameters to be set.
        """
        super().set_optimizable_parameters(params)

        for tone in self._envs:
            tone.set_optimizable_parameters(params)

    @partial(jit, static_argnums=(0,))
    def _compute_flat_top_envelope(self, t: Array) -> Array:
        t_final = self._t_final.get_value()
        ramp_time = t_final / 25
        ramp_up = 1 + erf((t - 2 * t_final / 20) / ramp_time)
        ramp_down = 1 + erf((-t + 18 * t_final / 20) / ramp_time)
        return ramp_up * ramp_down / 4

    def _evaluate(self, t: Array) -> Array:
        envelope: Array = self._compute_flat_top_envelope(t)
        return envelope

    @override
    def get_value(self, times: Array) -> Array:
        total_env: Array = jnp.zeros_like(times, dtype=jnp.complex128)
        for tone in self._envs:
            total_env += tone.get_value(times)

        flattop_env = self._compute_flat_top_envelope(times)
        total_env *= flattop_env
        return jnp.squeeze(total_env)

    @override
    def get_gradient(self, times: Array) -> Array:
        t_arr = jnp.array(times, ndmin=1)
        gradients = jnp.zeros(shape=(t_arr.shape[0], 0))
        total_env: Array = jnp.zeros_like(t_arr, dtype=jnp.complex128)
        smoothing = self._compute_flat_top_envelope(times)

        # Collect gradients wrt envelope parameters
        for tone in self._envs:
            value, grads = tone.get_value_and_gradient(times)
            total_env += value
            smoothing = jnp.reshape(smoothing, smoothing.shape + (1,) * (grads.ndim - smoothing.ndim))
            gradients = jnp.reshape(gradients, gradients.shape + (1,) * (grads.ndim - gradients.ndim))
            gradients = jnp.append(gradients, grads * smoothing, axis=1)
        return jnp.array(gradients)
