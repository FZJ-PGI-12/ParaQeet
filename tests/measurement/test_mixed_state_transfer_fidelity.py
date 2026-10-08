"""Test the mixed state transfer fidelity."""

import numpy as np
import pytest

from paraqeet.measurement.fidelity import Fidelity
from paraqeet.measurement.utils import densitiy_matrix_trace_fidelity, overlap_density_matrix_root
from tests.propagation.identity_propagation import IdentityPropagation
from tests.propagation.random_propagation import RandomPropagation


def random_mixed_state(dimension):
    """Generate random mixed states."""
    state = np.random.random((dimension, dimension)) + 1j * np.random.random((dimension, dimension))
    state = state @ np.conjugate(state.T)
    return state / np.trace(state)


def test_limits_vectors():
    """Test state vector limits.

    The fidelity for state vectors is always in the interval [0, 1).

    """
    for size in range(2, 30):
        target_state = random_mixed_state(size)
        propagation = RandomPropagation(size, True)
        times = np.array([1.0])
        measurement = Fidelity(
            propagation.get_value,
            propagation_and_gradient_func=None,
            target_states=target_state,
            overlap=overlap_density_matrix_root,
            fid=densitiy_matrix_trace_fidelity,
        )

        for i in range(100):
            m = measurement.get_value(times)
            assert 0.0 <= m <= 1.0


def test_vector_equality():
    """Test that F(v,v) = 1 for state vectors."""
    for size in range(2, 30):
        for i in range(100):
            state = random_mixed_state(size)
            propagation = IdentityPropagation()
            propagation.set_initial_state(state)
            measurement = Fidelity(
                propagation.get_value,
                propagation_and_gradient_func=None,
                target_states=state,
                overlap=overlap_density_matrix_root,
                fid=densitiy_matrix_trace_fidelity,
            )
            m = measurement.get_value(times=np.array([1.0]))
            np.testing.assert_almost_equal(m, 1.0, decimal=2)


def test_incompatible_shape():
    """Test incompatible shape of propagators.

    The propagators which have a different dimension than the
    target state should raise an exception.

    Raises
    ------
    Exception
        Different dimensions of the propagators raise an exception.

    """
    allDims = np.arange(2, 30)
    for dim in allDims:
        for i in range(100):
            target_state = random_mixed_state(dim)

            # create a propagator of a different dimension
            dimensions = np.delete(allDims, np.where(allDims == dim)[0][0])
            propagation = RandomPropagation(np.random.choice(dimensions), True)

            measurement = Fidelity(
                propagation.get_value,
                propagation_and_gradient_func=None,
                target_states=target_state,
                overlap=overlap_density_matrix_root,
                fid=densitiy_matrix_trace_fidelity,
            )

            with pytest.raises(Exception):
                measurement.get_value(times=np.array([1.0]))
