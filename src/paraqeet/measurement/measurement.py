"""Abstract measurement interfaces whose values serve as optimization goal functions."""

from abc import ABC, abstractmethod
from typing import Protocol

from paraqeet.quantity import Array, Float


class Measurement(ABC):
    """Represents any observable and the process of measurement itself.

    The observable is measured after the propagation class
    has solved the equation of motion.
    """

    @abstractmethod
    def get_value(self, times: Array) -> Array | Float:
        """Measure the observable and return the value.

        Args:
            times: One-dimensional vector of timestamps.

        Returns:
            This abstract method must return an Array or a Float when
            implemented by subclasses. Might return multiple values.
        """
        pass


class CostFunction(Measurement):
    """An abstract class for measurements providing normalized scalar value.

    Subclasses must implement the calculate_normalized_scalar() method which would
    return a measured value between 0 and 1.
    """

    @abstractmethod
    def calculate_normalized_scalar(self, times: Array) -> Float:
        """Measure the normalized observable.

        Returns a single scalar value between 0 and 1.
        This function must be implemented by subclasses.

        Args:
            times: One-dimensional vector of timestamps.

        Returns:
            Returns a Float if implemented by a subclass.
        """
        pass


class DifferentiableCostFunction(Protocol):
    """Protocol for a class that is both CostFuction and Differentiable.

    Note:
        Only used for type-hinting, not to be used as a base class.
    """

    def get_value(self, times: Array) -> Array | Float:
        """Return the value of the measurement."""
        ...

    def get_gradient(self, times: Array) -> Array:
        """Return the gradient of the measurement."""
        ...

    def get_value_and_gradient(self, times: Array) -> tuple[Array | Float, Array]:
        """Return the value and the gradient of the measurement."""
        ...

    def calculate_normalized_scalar(self, times: Array | Float) -> Float:
        """Usually the same as get_value."""
        ...
