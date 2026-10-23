"""Optimize a dCRAB pulse by a Scipy gradient based optimizer."""

import warnings
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
from scipy.optimize import OptimizeResult, minimize

from paraqeet.exceptions import ConfigurationException, IncompatibleOptimizationMap
from paraqeet.optimization_map import OptimizationMap
from paraqeet.optimizers.optimizer import OptimizationResult, Optimizer
from paraqeet.optimizers.scipy_optimizer import ScipyOptimizerGradient
from paraqeet.quantity import Array, Float, Quantity
from paraqeet.signal.envelopes import DCRABEnvelope
from paraqeet.signal.signal import DRAGMixer

warnings.simplefilter("once")


# Replace the default formatwarning
def custom_formatwarning(msg: Warning | str, *args: Any, **kwargs: Any) -> str:
    """Prettier warning statements."""
    return str(msg) + "\n"


warnings.formatwarning = custom_formatwarning  # type: ignore


class DCRABOptimizerGradient(ScipyOptimizerGradient):
    """A dCRAB optimization method :cite:p:`rach2015dressing,muller2022one`.

    Implements dCRAB optimization involving super-iterations that add additional
    optimization components to the dCRAB envelope and freezes the older parameters.

    Note:
        This works with a ``DCRABEnvelope`` or a ``list[DCRABEnvelope]`` as
        envelopes.
    """

    _num_iteration: int = 0
    _super_iteration_every: int
    _max_super_iteration_num: int
    _num_print_every: int
    _old_parameters_dict: dict[int, list[Quantity]]
    _old_opt_idxs_dict: dict[int, list[int]]

    best_params: Array | list[float]
    best_fid: float = 99999
    evaluations: int = 0

    _previous_fid: float = 99999
    _super_iteration_since: int = 0
    _super_iteration_num: int = 0
    _fallback_optimization: Optimizer | None
    _super_iteration_tol: float
    _seed: int | None = None

    def __init__(
        self,
        measure_func: Callable[[Array], Float],
        measure_and_gradient_func: Callable[[Array], tuple[Float, Array]],
        optimization_map: OptimizationMap,
        super_iteration_every: int = 30,
        max_super_iteration_num: int = 10,
        print_every_iteration_num: int = 5,
        fallback_optimization: Optimizer | None = None,
        super_iteration_tol: float = 1e-7,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            measure_and_gradient_func: Function implementing measurement of
                observables to be minimized, returning (value, gradient).
            optimization_map: An optimization map containing all parameters
                that can be optimized.
            super_iteration_every: Number of iterations after which one
                super-iteration is performed. Defaults to 30.
            max_super_iteration_num: Maximum number of super-iterations to
                perform. Defaults to 10.
            print_every_iteration_num: Print every this many iterations the
                current optimization value. Defaults to 5.
            fallback_optimization: Fallback optimization to perform after
                all super-iterations.
            super_iteration_tol: Tolerance for triggering a super-iteration.
                Defaults to 1e-7.
            seed: Random seed for adding new dCRAB components.
        """
        super().__init__(measure_func, measure_and_gradient_func, optimization_map)
        self._super_iteration_every = super_iteration_every
        self._max_super_iteration_num = max_super_iteration_num
        self._num_print_every = print_every_iteration_num
        self.callback = self._callback_function
        self._old_parameters_dict = {}
        self._old_opt_idxs_dict = {}
        self._fallback_optimization = fallback_optimization
        self._super_iteration_tol = super_iteration_tol
        self._seed = seed

    def _dcrab_super_iteration(self) -> None:
        optimizables = list(self._optimization_map.get_optimizables())
        relevant_optimizables = []

        dcrab_envs = []
        for opt in optimizables:
            if isinstance(opt, DCRABEnvelope):
                dcrab_envs.append(opt)
                relevant_optimizables.append(opt)
            else:
                try:
                    dcrab_envs.extend(opt.get_envelopes())  # type:ignore
                    relevant_optimizables.append(opt)  # type:ignore
                except Exception as e:
                    raise ConfigurationException(
                        "Optimizable is not a ``DCRABEnvelope`` nor does it have a ``get_envelope`` method.\n"
                        + f"Raised exception {e}"
                    )

        # Add new parameters to the dCRAB envelope
        for n, env in enumerate(dcrab_envs):
            try:
                if self._seed is not None:
                    seed = (n + 1) * self._super_iteration_num * self._seed
                else:
                    seed = None
                env.add_new_components(seed=seed)
            except Exception as e:
                raise ConfigurationException(f"Non-``DCRABEnvelope`` encountered. \n Raised exception {e}")

        # Add new parameters to optmap
        new_coeffs_freqs_and_phases = [env.get_coefficients_frequencies_and_phases() for env in dcrab_envs]
        for env, coeffs_freqs_phases in zip(relevant_optimizables, new_coeffs_freqs_and_phases):
            params = [env.get_parameters()[0]] + coeffs_freqs_phases  # Add pulse amplitude to optimization
            if isinstance(env, DRAGMixer):
                params.append(env.get_parameters()[-1])  # Add pulse delta to optimization if it is a DRAG tone.
            self._optimization_map.add(env, params)

        # Get all parameters and update scales for optimization
        params = self._optimization_map.get_all_parameters()
        self._scales = jnp.array([p.get_scale() for p in params]).flatten()
        print(f"* Current no. of parameters = {len(params)}")

        # Restart the optimization process
        self._build_optimizable_index_list()
        self._optimization_map.register_params_with_optimizables()

        init = []
        for qty in self._optimization_map.get_all_parameters():
            init.append(qty.get_reduced_value())

        self._minimize_infidelity(init)

    def _callback_function(self, intermediate_result: OptimizeResult) -> None:
        if self._num_iteration % self._num_print_every == 0:
            print(f"Iteration number = {self._num_iteration} \t  Infidelity  = {intermediate_result.fun:.3e}")

        self._num_iteration += 1
        self._super_iteration_since += 1

        if intermediate_result.fun < self.best_fid:
            self.best_fid = intermediate_result.fun
            self.best_params = intermediate_result.x

        if self._super_iteration_num >= self._max_super_iteration_num:
            raise StopIteration("Maximum number of super iterations performed. Stopping optimization.")

        if np.abs(intermediate_result.fun - self._previous_fid) < self._super_iteration_tol:
            self._super_iteration_num += 1

            print("\n")
            print(f"==== Decrease in infidelity less than {self._super_iteration_tol} ====")
            print(f"==== Starting super-iteration {self._super_iteration_num} ====")
            print(f"* Current lowest infidelity = {self.best_fid: .3e}")

            current_params = self._optimization_map.get_all_parameters()
            self._old_parameters_dict[len(current_params)] = current_params
            self._old_opt_idxs_dict[len(current_params)] = self._opt_idxs

            # Start the dCRAB super-iteration
            self._super_iteration_since = 0
            self._dcrab_super_iteration()

        self._previous_fid = intermediate_result.fun

        if self._super_iteration_since >= self._super_iteration_every:
            self._super_iteration_num += 1

            print("\n")
            print("==== Max iteration before a super-iteration reached ====")
            print(f"==== Starting super-iteration {self._super_iteration_num} ====")
            print(f"*** Current lowest infidelity = {self.best_fid: .3f} ***")

            current_params = self._optimization_map.get_all_parameters()
            self._old_parameters_dict[len(current_params)] = current_params
            self._old_opt_idxs_dict[len(current_params)] = self._opt_idxs

            # Start the dCRAB super-iteration
            self._super_iteration_since = 0
            self._dcrab_super_iteration()

    def set_parameters(self, values: Array | list[float]) -> list[Quantity]:
        """Update the parameter values.

        This method is derived from the ``ScipyOptimizerGradient`` class and
        designed to catch cases involving mismatch in dimension of parameters.

        Since, in dCRAB, new parameters are added in each super-iteration,
        the previous best result may be one with fewer parameters.
        In that case, all subsequent parameters are set to their minimum value
        (by setting the reduced value to -1).

        Args:
            values: Parameter values for the update.

        Returns:
            The log of parameter updates.
        """
        log = []
        params = self._optimization_map.get_all_parameters()

        # Check for edge case when best result has fewer parameters than the current iteration.
        if len(params) != len(values):
            num_values = len(values)
            warnings.warn(
                "Stopping the optimization or backtracking to previous best fidelity."
                + f" Going to step with {num_values} parameters."
            )
            if num_values in self._old_parameters_dict:
                ids = [id(i) for i in self._old_parameters_dict[num_values]]
                j = 0
                for ii in range(len(params)):
                    if id(params[ii]) in ids:
                        if j >= num_values:
                            raise Exception("Possible duplicate entries in the optmap.")
                        params[ii].set_reduced_value(np.array([values[j]]))
                        j += 1
                    else:
                        params[ii].set_reduced_value(np.array([0]))
                    log.append(params[ii])
            else:
                raise ConfigurationException(
                    f"Got values from optimization than does not fit in optimization map. \n \
                    No. of values from optimization = {num_values} \
                    and no. of objects in optimization map \
                    (in the different super-iterations) = {list(self._old_parameters_dict.keys())}."
                )
        else:
            for index, val in enumerate(np.split(values, self._opt_idxs[:-1])):
                params[index].set_reduced_value(val)
                log.append(params[index])
        return log

    def _set_parameters_and_measure(self, values: Array) -> float:
        """Update the parameter values and return measurement result.

        Args:
            values: Parameter values for the update.

        Returns:
            The infidelity, i.e. one minus the fidelity (the optimizer minimizes this).
        """
        log = self.set_parameters(values)

        fun, grad = self._measure_and_gradient_func(self._times)
        self._grad_cache = grad

        infid = 1.0 - fun
        if self._logger:
            self._logger.log(log, float(infid))
        self.evaluations += 1
        return float(1 - fun)

    def _minimize_infidelity(self, init: list[Array]) -> OptimizeResult:
        result = minimize(
            fun=self._set_parameters_and_measure,
            jac=self._lookup_jac,
            x0=jnp.concatenate(init).flatten(),
            bounds=[(-1, 1)] * self._opt_idxs[-1],
            method=self._method,
            options=self._options,
            callback=self._callback,
        )
        return result

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

        self.best_params = list(jnp.concatenate(init).flatten())

        try:
            result = self._minimize_infidelity(init)
        except Exception as e:
            if "_lbfgsb._lbfgsb.setulb: failed to create array from the 7th" + " argument ``g``" in str(e):
                raise IncompatibleOptimizationMap(
                    "Number of quantities in optMap differ from number of" + f" gradients computed. \n {e}"
                )
            else:
                raise e

        # Set the parameters to the best params at the end of optimization
        print("Setting parameters to the best values.")
        self.set_parameters(self.best_params)

        if self._fallback_optimization is not None:
            print("Performing fallback optimization")
            self._fallback_optimization.optimize(self._times)

        if self._logger:
            self._logger.stop(str(result))

        return OptimizationResult(
            status=(OptimizationResult.STATUS_SUCCESS if result.success else OptimizationResult.STATUS_FAILED),
            value=self.best_fid,
            iterations=self._num_iteration,
            evaluations=self.evaluations,
            message=result.message,
            raw_result=result,
        )
