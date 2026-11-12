"""Class definition of the Weighted Sum Goal model."""

import itertools
from typing import override

import jax.numpy as jnp
import numpy as np

from paraqeet.differentiable import Differentiable
from paraqeet.exceptions import ConfigurationException
from paraqeet.measurement.measurement import CostFunction, DifferentiableCostFunction
from paraqeet.quantity import Array, Float


class WeightedSumGoal(CostFunction, Differentiable):
    """Combine multiple measurements into a single goal function."""

    _measurements: list[DifferentiableCostFunction]
    _weights: Array
    _sum_of_squares_options: dict | None
    _measurements_in_sum_of_squares: list[DifferentiableCostFunction]

    def __init__(
        self,
        measurements: list[DifferentiableCostFunction],
        weights: Array,
        sum_of_squares_options: dict | None = None,
    ) -> None:
        """
        Args:
            measurements: List of measurements.
            weights: List of weights.
            sum_of_squares_options: A dictionary that contains information
                about how to include the sum of square differences in the
                cost function. If not None then it must contain the keys:

                - ``weight`` (float): The weight of the sum of square differences.
                - ``meas_bool`` (list[bool]): A list of booleans of the same
                  length as measurements. If an element is True, the
                  corresponding measurement is included in the sum of square
                  difference goal function.

        Raises:
            ConfigurationException: If number of measurements and weights
                are incompatible.
            UserWarning: If the given weights are not normalized.
        """
        self._measurements = measurements
        self._weights = weights
        self._sum_of_squares_options = sum_of_squares_options
        if len(measurements) != len(weights):
            raise ConfigurationException(
                f"Incompatible number of measurements {len(measurements)} and weights {len(weights)}"
            )
        if sum_of_squares_options is not None:
            expected_keys = ["weight", "meas_bool"]
            provided_keys = list(sum_of_squares_options.keys())

            for key in provided_keys:
                if key not in expected_keys:
                    raise KeyError("The dictionary must have weight and meas_bool as keys")

            if not np.isclose(sum(weights) + sum_of_squares_options["weight"], 1.0):
                raise UserWarning("Total supplied weights are not normalized.")

            self._measurements_in_sum_of_squares = [
                meas for meas, flag in zip(measurements, sum_of_squares_options["meas_bool"]) if flag
            ]
        else:
            if not np.isclose(sum(weights), 1.0):
                raise UserWarning("Supplied weights are not normalized.")
            self._measurements_in_sum_of_squares = []

        for meas in self._measurements:
            if not isinstance(meas, Differentiable):
                raise ConfigurationException(
                    "All measurements must be Differentiable to compute the gradient of the WeightedSumGoal"
                )

    @property
    def measurements(self) -> list[DifferentiableCostFunction]:
        """Return the list of measurements."""
        return self._measurements

    @property
    def weights(self) -> Array:
        """Return the weights used in the weighted goal function."""
        return self._weights

    @property
    def sum_of_square_options(self) -> dict | None:
        """Return the dictionary with the options about the sum of square differences goal function."""
        return self._sum_of_squares_options

    @property
    def measurements_in_sum_of_squares(self) -> list[DifferentiableCostFunction]:
        """Return the list of measurements included in the sum of square difference cost function."""
        return self._measurements_in_sum_of_squares

    @override
    def get_value(self, times: Array) -> Array | Float:
        """Sum of plain weighted measurements.

        Returns:
            Returns the plain weighted sum.
        """
        values = [m.get_value(times=times) for m in self._measurements]
        sum_meas: Array | Float = 0.0
        for ii, w in enumerate(self._weights):
            sum_meas += w * values[ii]
        if self._sum_of_squares_options is not None:
            sum_square_diff: Array | Float = 0.0
            values_in_sum_of_squares = [
                val for val, flag in zip(values, self._sum_of_squares_options["meas_bool"]) if flag
            ]
            for meas_a, meas_b in itertools.combinations(values_in_sum_of_squares, 2):
                sum_square_diff += (meas_a - meas_b) ** 2
            sum_meas += self._sum_of_squares_options["weight"] * sum_square_diff
        return sum_meas

    @override
    def calculate_normalized_scalar(self, times: Array) -> Float:
        return jnp.array(self.get_value(times))

    @override
    def get_value_and_gradient(self, times: Array) -> tuple[Array, Array]:
        """Sum of weighted measurements from gradient-ized measurements.

        Args:
            times: Array of times.

        Returns:
            The sum of gradients.
        """
        values_and_gradients = [m.get_value_and_gradient(times=times) for m in self._measurements]
        sum_meas = jnp.array(0)
        sum_grads = jnp.zeros_like(values_and_gradients[0][1])
        for ii, w in enumerate(self._weights):
            sum_meas += w * values_and_gradients[ii][0]
            sum_grads += w * values_and_gradients[ii][1]
        if self._sum_of_squares_options is not None:
            values_and_gradients_in_sum_of_squares = [
                val_grad
                for val_grad, flag in zip(values_and_gradients, self._sum_of_squares_options["meas_bool"])
                if flag
            ]
            sum_square_diff = jnp.array(0)
            grads_diff = jnp.zeros_like(values_and_gradients_in_sum_of_squares[0][1])

            for meas_a, meas_b in itertools.combinations(values_and_gradients_in_sum_of_squares, 2):
                sum_square_diff += (meas_a[0] - meas_b[0]) ** 2
                grads_diff += 2 * (meas_a[0] - meas_b[0]) * (meas_a[1] - meas_b[1])
            sum_meas += self._sum_of_squares_options["weight"] * sum_square_diff
            sum_grads += self._sum_of_squares_options["weight"] * grads_diff
        return sum_meas, sum_grads
