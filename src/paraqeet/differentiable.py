"""Base class for all classes that provide gradients."""

from abc import ABC, abstractmethod

from paraqeet.quantity import Array, Float


class Differentiable(ABC):
    """An abstract class for differentiable models.

    Subclasses must implement the get_value, get_gradient.
    Note that in some cases the gradient gets computed together with the
    value and thus it is more natural to instantiate a get_gradient method
    using the result of get_value_and_gradient. In these cases, it makes sense
    to override the get_value_and_gradient method.
    """

    @abstractmethod
    def get_value(self, times: Array) -> Array | Float:
        """Calculate the value of the object.

        Args:
            times: Array of times.

        Returns:
            The value of the object.
            If it returns an Array then the value is calculated at the n_times and the dimension should be
            (n_times, (dimensions_of_object)). If the object is a scalar (1x1 Array)
            the dimension is just n_times.
            If it returns a Float for instance it means that the object depends on the whole
            array of times. This is for instance the case of fidelities
            that are a function of an array of times.
        """
        pass

    def get_gradient(self, times: Array) -> Array:
        """Calculate the gradient of the object.

        Args:
            times: Array of times.

        Returns:
            The gradient of the object. There are two main cases.

            1) The array has dimensions (n_times, n_params, (dimensions_of_object)).
            If the object is a scalar (1x1 Array) the dimension is just (n_times, n_params).

            2) The array has dimension (n_params, (dimensions_of_object)). This is the case for instance of fidelities
            that are a function of an array of times.
        """
        _, gradient = self.get_value_and_gradient(times)
        return gradient

    def get_value_and_gradient(self, times: Array) -> tuple[Array | Float, Array]:
        """Calculate the value and the gradient of the object.

        Note:
            The default implementation here gathers the value and the gradient separately.
            For cases where the value can be obtained during the gradient calculation,
            this method is overwritten for efficiency.

        Returns:
            The value and the gradient of the object.
        """
        return self.get_value(times), self.get_gradient(times)
