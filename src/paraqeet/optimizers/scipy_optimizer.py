"""Class definition of the Scipy optimizer model."""

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
from scipy.optimize import OptimizeResult, minimize

from paraqeet.exceptions import IncompatibleOptimizationMap
from paraqeet.optimization_map import OptimizationMap
from paraqeet.optimizers.optimizer import OptimizationResult, Optimizer
from paraqeet.quantity import Array, Float


class ScipyOptimizer(Optimizer):
    """Minimize the outcome of a measurement with the scipy optimization package."""

    _measure_func: Callable[[Array], Float]
    _opt_idxs: list[int]
    _options: dict
    _method: str
    _callback: Callable | None
    _num_iterations: int = 0

    def __init__(self, measure_func: Callable[[Array], Float], optimization_map: OptimizationMap) -> None:
        """
        Args:
            measure_func: Function implementing measurement of observables
                to be minimized.
            optimization_map: An optimization map containing all parameters
                that can be optimized.
        """
        super().__init__(measure_func, optimization_map)
        self._options = {}
        self._method = "L-BFGS-B"
        self._callback = self._default_callback

    @property
    def method(self) -> str:
        """Returns the currently selected optimization method."""
        return self._method

    @method.setter
    def method(self, method: str) -> None:
        """Select method from scipy.optimize.minimize.

        Args:
            method: Type of solver, specified by string value.
        """
        self._method = method

    def set_options(self, opts: dict) -> None:
        """Set the options for the system."""
        self._options.update(opts)

    def update_option(self, key: str, val: Any) -> None:
        """Update one option for the system."""
        self._options[key] = val

    @property
    def callback(self) -> Callable | None:
        """Returns the callback function."""
        return self._callback

    @callback.setter
    def callback(self, cbfun: Callable) -> None:
        """Set the callback function for the optimizer.

        Args:
            cbfun: The function to be set as the callback.
        """
        self._callback = cbfun

    def _default_callback(self, intermediate_result: OptimizeResult) -> None:
        self._num_iterations += 1
        fun = intermediate_result.fun if hasattr(intermediate_result, "fun") else None
        if self._num_iterations % 10 == 0:
            print(f"Iteration {self._num_iterations:4d} | Infid = {fun:.6e}")

    def optimize(self, times: Array | float) -> OptimizationResult:
        """Optimize the system via the Scipy optimizer.

        Performs the actual optimization.

        Since the search parameters are dimensionless and bound by [-1, 1],
        we set the bounds of the scipy minimize module to -1 and 1 explicitly
        in each search dimension.

        Note:
            If input ``times`` is a float, then the start time of propagation
            is implicitly assumed to be zero. For an array of times, the first
            time point is the start time.

        Args:
            times: Array of times or a float (assumed start time zero).

        Returns:
            The result of the optimization.
        """
        if self._logger:
            self._logger.start()

        self._times = jnp.array([0.0, times]) if isinstance(times, float) else times

        self._build_optimizable_index_list()
        self._optimization_map.register_params_with_optimizables()

        # Collect the initial values of all parameters
        init = []
        for qty in self._optimization_map.get_all_parameters():
            init.append(qty.get_reduced_value())  # reduced values are between [-1, 1]

        opt_res = minimize(
            fun=self._set_parameters_and_measure,
            x0=np.concatenate(init).flatten(),
            bounds=[(-1, 1)] * self._opt_idxs[-1],
            method=self._method,
            options=self._options,
            callback=self._callback,
        )

        if self._logger:
            self._logger.stop(str(opt_res))

        return OptimizationResult(
            status=OptimizationResult.STATUS_SUCCESS if opt_res.success else OptimizationResult.STATUS_FAILED,
            value=opt_res.fun,
            iterations=opt_res.nit,
            evaluations=opt_res.nfev,
            message=opt_res.message,
            raw_result=opt_res,
        )

    def _set_parameters_and_measure(self, values: Array) -> Float:
        """Update the parameter values and return the measurement result.

        Internal callback.

        Args:
            values: Parameter values for the update.

        Returns:
            The infidelity, i.e. one minus the measurement result (the optimizer minimizes this).
        """
        log = []
        params = self._optimization_map.get_all_parameters()
        for index, val in enumerate(np.split(values, self._opt_idxs[:-1])):
            params[index].set_reduced_value(val)
            log.append(params[index])
        infid = 1 - self._measure_func(self._times)

        if self._logger:
            self._logger.log(log, float(infid))
        return infid


class ScipyOptimizerGradient(ScipyOptimizer):
    """The Scipy Optimizer gradient model.

    Minimize the outcome of a measurement with the Scipy optimization package
    by providing gradient values.
    """

    _grad_cache: Array  # of shape (n_parameters,)
    _scales: Array
    _measure_func: Callable[[Array], Float]
    _measure_and_gradient_func: Callable[[Array], tuple[Float, Array]]

    def __init__(
        self,
        measure_func: Callable[[Array], Float],
        measure_and_gradient_func: Callable[[Array], tuple[Float, Array]],
        optimization_map: OptimizationMap,
    ) -> None:
        """
        Args:
            measure_and_gradient_func: Function implementing measurement of
                observables to be minimized, returning (value, gradient).
            optimization_map: An optimization map containing all parameters
                that can be optimized.
        """
        self._measure_and_gradient_func = measure_and_gradient_func
        super().__init__(measure_func, optimization_map)
        params = self._optimization_map.get_all_parameters()
        self._scales = jnp.array([p.get_scale() for p in params]).flatten()

    def optimize(self, times: Array | float) -> OptimizationResult:
        """Optimize via the Scipy optimizer gradient model.

        Performs the actual optimization.

        Note:
            If input ``times`` is a float, then the start time of propagation
            is implicitly assumed to be zero. For an array of times, the first
            time point is the start time.

        Args:
            times: Array of times or a float (assumed start time zero).

        Returns:
            The result of the optimization.
        """
        self._times = jnp.array([0.0, times]) if isinstance(times, float) else times

        if self._logger:
            self._logger.start()

        self._build_optimizable_index_list()
        self._optimization_map.register_params_with_optimizables()

        init = []
        for qty in self._optimization_map.get_all_parameters():
            init.append(qty.get_reduced_value())

        try:
            result = minimize(
                fun=self._set_parameters_and_measure,
                jac=self._lookup_jac,
                x0=jnp.concatenate(init).flatten(),
                bounds=[(-1, 1)] * self._opt_idxs[-1],
                method=self._method,
                options=self._options,
                callback=self._callback,
            )
        except Exception as e:
            if "_lbfgsb._lbfgsb.setulb: failed to create array from the 7th" + " argument ``g``" in str(e):
                raise IncompatibleOptimizationMap(
                    "Number of quantities in optMap differ from number of" + f" gradients computed. \n {e}"
                )
            else:
                raise e

        if self._logger:
            self._logger.stop(str(result))

        return OptimizationResult(
            status=(OptimizationResult.STATUS_SUCCESS if result.success else OptimizationResult.STATUS_FAILED),
            value=result.fun,
            iterations=result.nit,
            evaluations=result.nfev,
            message=result.message,
            raw_result=result,
        )

    def _set_parameters_and_measure(self, values: Array) -> float:
        """Update the parameter values and return measurement result.

        Returns the measurement result including gradient.
        The gradient is stored in a local cache for lookup.
        This is tailored for L-BFGS-B or similar algorithms that alternate
        between function and gradient calls.
        Internal callback.

        Args:
            values: Parameter values for the update.

        Returns:
            The infidelity, i.e. one minus the fidelity (the optimizer minimizes this).
        """
        log = []
        params = self._optimization_map.get_all_parameters()
        for index, val in enumerate(np.split(values, self._opt_idxs[:-1])):  # TODO: Convert to jax
            params[index].set_reduced_value(val)
            log.append(params[index])

        fun, grad = self._measure_and_gradient_func(self._times)
        self._grad_cache = grad

        infid = 1.0 - fun
        if self._logger:
            self._logger.log(log, float(infid))
        return float(1 - fun)

    def _lookup_jac(self, values: Array) -> Array:
        """Update the parameter values.

        Return the gradient of a measurement result.
        Internal callback.

        Args:
            values: Parameter values for the update.

        Returns:
            The gradient of a measurement result.
        """
        return -1 * self._grad_cache * self._scales
