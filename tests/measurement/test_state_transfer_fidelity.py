"""Test the state transfer fidelity model."""

import numpy as np
import pytest

from paraqeet.measurement.fidelity import Fidelity
from paraqeet.measurement.utils import overlap_state_vector, state_fidelity
from tests.propagation.identity_propagation import IdentityPropagation
from tests.propagation.random_propagation import RandomPropagation


@pytest.fixture
def identity_propagation():
    """Return a mock identity propagation object."""
    return IdentityPropagation()


def test_limits_vectors(random_state):
    """Test fidelity for state vectors is always in the interval [0, 1)."""
    for size in range(2, 30):
        target_state = random_state(size)
        propagation = RandomPropagation(size, False)
        times = np.array([0.0, 1.0])
        measurement = Fidelity(
            propagation_func=propagation.get_value,
            propagation_and_gradient_func=propagation.get_value_and_gradient,
            target_states=target_state,
            overlap=overlap_state_vector,
            fid=state_fidelity,
        )

        for _ in range(20):
            m = measurement.get_value(times=times)
            assert 0.0 <= m
            m = measurement.calculate_normalized_scalar(times=times)
            assert 0.0 <= m <= 1.0
            m, _ = measurement.get_value_and_gradient(times=times)
            assert 0.0 <= m


def test_vector_equality(identity_propagation, random_state):
    """Test that F(v,v) = 1 for state vectors."""
    for size in range(2, 30):
        for _ in range(100):
            state = random_state(size)
            identity_propagation.set_initial_state(state)
            measurement = Fidelity(
                propagation_func=identity_propagation.get_value,
                propagation_and_gradient_func=identity_propagation.get_gradient,
                target_states=state,
                overlap=overlap_state_vector,
                fid=state_fidelity,
            )
            m = measurement.get_value(times=np.array([1.0]))
            np.testing.assert_almost_equal(m, 1.0)


@pytest.mark.filterwarnings("ignore:Different shapes for")
def test_incompatible_shape(identity_propagation, random_state):
    """Test incompatible shapes for initial and target states.

    A set of initial and target states with different dimensions
    should raise an exception.

    Raises
    ------
    Exception
        Different dimensions for the initial and target states
        raise an exception.

    """
    allDims = np.arange(2, 30)
    for dim in allDims:
        for i in range(10):
            initialState = random_state(dim)
            dimensions = np.delete(allDims, np.where(allDims == dim)[0][0])
            targetState = random_state(np.random.choice(dimensions))

            identity_propagation.set_initial_state(initialState)

            fid = Fidelity(
                propagation_func=identity_propagation.get_value,
                propagation_and_gradient_func=identity_propagation.get_gradient,
                target_states=targetState,
                overlap=overlap_state_vector,
                fid=state_fidelity,
            )

            with pytest.raises(Exception):
                fid.get_value(times=np.array([1.0]))
