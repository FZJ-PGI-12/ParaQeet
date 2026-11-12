"""Test the unitary fidelity model."""

import numpy as np
import pytest

from paraqeet import UnitaryFidelity
from tests.propagation.identity_propagation import IdentityPropagation
from tests.propagation.random_propagation import RandomPropagation


@pytest.fixture
def identity_propagation():
    """Return a mock identity propagation object."""
    return IdentityPropagation()


def test_positivity(random_unitary_matrix, random_basis_vectors):
    """Test that the fidelity is always positive."""
    for dim in range(5, 10):
        for i in range(50):
            propagation = RandomPropagation(dim, True)
            gate = random_unitary_matrix(dim)
            measurement = UnitaryFidelity(
                propagation_func=propagation.get_value,
                propagation_and_gradient_func=propagation.get_gradient,
                gate=gate,
            )
            times = np.array([1.0])
            m = measurement.get_value(times=times)
            assert 0.0 <= m

            # gate must have at least two dimensions
            subDim = np.random.randint(2, dim)
            basisStates = random_basis_vectors(dim, subDim)
            # gate is defined on the subspace.
            gate = random_unitary_matrix(subDim)
            measurement = UnitaryFidelity(
                propagation_func=propagation.get_value,
                propagation_and_gradient_func=propagation.get_gradient,
                gate=gate,
                basis_states=basisStates,
            )
            m = measurement.get_value(times=times)
            assert 0.0 <= m


def test_equality(identity_propagation, random_unitary_matrix):
    """Test that F(U,U) = 1."""
    for dim in range(2, 10):
        for i in range(50):
            gate = random_unitary_matrix(dim)
            np.testing.assert_almost_equal(np.conjugate(gate.T) @ gate, np.eye(dim))
            propagation = IdentityPropagation()
            propagation.set_initial_state(gate)
            measurement = UnitaryFidelity(
                propagation_func=propagation.get_value,
                propagation_and_gradient_func=propagation.get_gradient,
                gate=gate,
            )
            m = measurement.get_value(times=np.array([1.0]))
            np.testing.assert_almost_equal(m, 1.0)


@pytest.mark.skip(reason="Projection needed")
def test_projection(identity_propagation, random_basis_vectors):
    """Test the projection of the state vectors."""
    for dim in range(2, 10):
        for i in range(50):
            gate = np.eye(dim)
            init_state = random_basis_vectors(dim + 4, dim)
            propagation = identity_propagation
            propagation.set_initial_state(gate)
            measurement = UnitaryFidelity(
                propagation_func=propagation.get_value,
                propagation_and_gradient_func=propagation.get_gradient,
                gate=gate,
                basis_states=init_state,
            )
            m = measurement.get_value(times=np.array([1.0]))
            np.testing.assert_almost_equal(m, 1.0)


def test_incompatible_shape(identity_propagation, random_unitary_matrix):
    """Test incompatible shapes for initial and target states.

    A set of initial and target states with different dimensions
    should raise an exception.

    Raises
    ------
    Exception
        Different dimensions for the initial and target states
        raise an exception.

    """
    allDims = np.arange(2, 10)
    for dim in allDims:
        for i in range(50):
            gate = random_unitary_matrix(dim)

            # create a propagator of a different dimension
            dimensions = np.delete(allDims, np.where(allDims == dim)[0][0])
            propagation = RandomPropagation(np.random.choice(dimensions), True)

            measurement = UnitaryFidelity(
                propagation_func=propagation.get_value,
                propagation_and_gradient_func=propagation.get_gradient,
                gate=gate,
            )
            with pytest.raises(Exception):
                measurement.get_value(times=np.array([1.0]))
