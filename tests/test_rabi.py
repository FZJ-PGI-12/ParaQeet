"""Testing the Rabi Analytic Model."""

import jax
import pytest
from numpy.testing import assert_almost_equal

from paraqeet.measurement.fidelity import Fidelity
from paraqeet.measurement.utils import overlap_state_vector, state_fidelity
from paraqeet.optimization_map import OptimizationMap
from paraqeet.optimizers.scipy_optimizer import ScipyOptimizer
from paraqeet.propagation.rabi_model import RabiModel

jax.config.update("jax_enable_x64", True)

FREQ = 4.8e9
RABI_NAME = "Analytic Rabi Model"
T_FINAL = 0.6e-9


@pytest.fixture
def opt(rabi):
    """Create optimization map with Rabi model."""
    optmap = OptimizationMap()
    optmap.add(rabi, rabi.get_parameters())
    fid = Fidelity(
        propagation_func=rabi.get_value,
        propagation_and_gradient_func=None,
        target_states=jax.numpy.array([[0.0], [1]]),
        fid=state_fidelity,
        overlap=overlap_state_vector,
    )
    return ScipyOptimizer(measure_func=fid.calculate_normalized_scalar, optimization_map=optmap)


@pytest.fixture
def rabi():
    """Create Rabi test object."""
    exp = RabiModel(FREQ)
    exp.name = RABI_NAME
    return exp


def test_name(rabi):
    """Check that the name is 'RABI_NAME'."""
    assert rabi.name == RABI_NAME


def test_rabi(opt) -> None:
    """Check that the rabi optimization goes below threshold."""
    res = opt.optimize(times=T_FINAL)
    assert res.value < 1e-7


def test_find_resonance(rabi, opt) -> None:
    """Check for resonance."""
    opt.optimize(times=T_FINAL)
    params = rabi.get_parameters()
    assert_almost_equal(FREQ / 1e9, params[1].get_value() / 1e9, decimal=4)
