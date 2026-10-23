"""Test GOAT over GRAPE. This is same as the example OC_GOAToverGRAPE_TLS"""

from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
import numpy as np
import numpy.testing as testing
import pytest
from jax import jit
from jax.scipy.special import erf

from paraqeet.eom.schroedinger_equation import SchroedingerEquation
from paraqeet.hamiltonian.drive import Drive
from paraqeet.hamiltonian.qubit import QubitHamiltonian
from paraqeet.measurement.fidelity import (
    FidelityGRAPE,
)
from paraqeet.measurement.goat_over_grape import GOATOverGRAPE
from paraqeet.measurement.utils import overlap_state_vector, state_fidelity
from paraqeet.optimization_map import OptimizationMap
from paraqeet.optimizers.scipy_optimizer import ScipyOptimizerGradient
from paraqeet.propagation.expm import Expm
from paraqeet.propagation.grape import GRAPE
from paraqeet.propagation.utils import grape_operator_sandwich_function_closed
from paraqeet.quantity import Array, Quantity
from paraqeet.signal.envelopes import Envelope
from paraqeet.signal.pwc_generator import PWCGenerator

T_FINAL = 20e-9
TLIST = jnp.linspace(0, T_FINAL, 26)


class FlatTopGaussianEnvelope(Envelope):
    """A flat-top Gaussian envelope."""

    def __init__(
        self,
        amplitude: Quantity,
        t_up: Quantity,
        t_down: Quantity,
        ramp_time: Quantity,
    ):
        self._amplitude = amplitude
        self._t_up = t_up
        self._t_down = t_down
        self._ramp_time = ramp_time

        self._gradient_function: Callable | None = None
        self._grad_arg_nums: tuple[int, ...] = ()

    def get_parameters(self):
        """Get all parameters of the system."""
        return [self._amplitude, self._t_up, self._t_down, self._ramp_time]

    @partial(jit, static_argnums=(0,))
    def _evaluate(self, amp: Array, t_up: Array, t_down: Array, ramp_time: Array, t: Array):
        ramp_up = 1 + erf((t - t_up) / ramp_time)
        ramp_down = 1 + erf((-t + t_down) / ramp_time)
        return jnp.squeeze(amp * ramp_up * ramp_down / 4)

    def get_value(self, t: Array | float) -> Array:
        """Compute pulse shape."""
        amp = self._amplitude.get_value()
        t_up = self._t_up.get_value()
        t_down = self._t_down.get_value()
        ramp_time = self._ramp_time.get_value()
        # returns JitWrapped
        return self._evaluate(amp, t_up, t_down, ramp_time, t)  # type: ignore


@pytest.fixture
def tone():
    tone = FlatTopGaussianEnvelope(
        amplitude=Quantity(jnp.pi / T_FINAL / 3, -jnp.pi / T_FINAL, jnp.pi / T_FINAL, name="Amplitude"),
        t_up=Quantity(1e-9, 0.0, T_FINAL, name="t_up"),
        t_down=Quantity(T_FINAL - 1e-9, 0.0, T_FINAL, name="t_down"),
        ramp_time=Quantity(2e-9, 0.5e-9, T_FINAL, name="ramp_time"),
    )
    return tone


@pytest.fixture
def gen(tone):
    gen = PWCGenerator(envelopes=[tone], tlist=TLIST)
    gen.multiply_flat_top = True

    params = gen.get_parameters()
    params[0].set_limits(-200e6, 200e6)
    params[1].set_limits(-200e6, 200e6)
    return gen


@pytest.fixture
def model(gen):
    qubit_hamiltonian = QubitHamiltonian(frequency=Quantity(0.0, 0.0, 2 * np.pi * 1e6, unit="Hz"), drives=[])
    drive = Drive(qubit_hamiltonian.sigma_minus, gen, add_hermitian=True)
    qubit_hamiltonian.drives = [drive]
    model = SchroedingerEquation(
        hamiltonian_func=qubit_hamiltonian.get_value,
        hamiltonian_gradient_func=qubit_hamiltonian.get_gradient,
    )

    return model


@pytest.fixture
def propagation(model):
    init = jnp.array([[1.0], [0]])  # |0>
    return Expm(eom_func=model.get_value, resolution=1e9, initial_state=init)


@pytest.fixture
def prop(model, propagation):
    target = jnp.array([[0.0], [1]])  # |1>

    prop = GRAPE(
        propagation,
        eom_gradient_func=model.get_gradient,
        target_state=target,
        operator_sandwich_function=grape_operator_sandwich_function_closed,
        order=3,
    )
    return prop


@pytest.fixture
def fid(prop):
    target = jnp.array([[0.0], [1.0]])  # |1>

    zeroone = FidelityGRAPE(
        propagation_func=prop.get_value,
        propagation_and_gradient_func=prop.get_value_and_gradient,
        overlap=overlap_state_vector,
        fid=state_fidelity,
        target_states=target,
    )
    return zeroone


@pytest.fixture
def opt_grad(tone, fid, gen, propagation):
    optmap = OptimizationMap()
    optmap.add(tone)
    optmap.register_params_with_optimizables()

    goat = GOATOverGRAPE(fid, generators=gen, propagation_resolution=propagation.resolution)
    opt_grad = ScipyOptimizerGradient(measure_and_gradient_func=goat.get_value_and_gradient, optimization_map=optmap)
    return opt_grad


def test_can_measure(fid, gen, propagation):
    fid = GOATOverGRAPE(fid, generators=[gen], propagation_resolution=propagation.resolution)
    val, grad = fid.get_value_and_gradient(times=TLIST)
    assert 0 <= fid.get_value(times=TLIST)
    assert 0 <= val <= 1

    value = fid.calculate_normalized_scalar(times=TLIST)
    assert 0 <= value
    testing.assert_array_less(np.zeros_like(grad), grad)


def test_goat_over_grape(opt_grad):
    res = opt_grad.optimize(times=TLIST)
    assert res.value < 1e-4


def test_gradient_on_a_grid_that_differs_from_the_pixels(tone, fid, gen, propagation):
    """Test that the gradients are computed correctly for a different time grid."""
    optmap = OptimizationMap()
    optmap.add(tone)
    optmap.register_params_with_optimizables()
    goat = GOATOverGRAPE(fid, generators=gen, propagation_resolution=propagation.resolution)
    times = jnp.array([0.0, T_FINAL])

    _, gradient = goat.get_value_and_gradient(times=times)

    finite_differences = []
    for parameter in tone.get_parameters():
        value = parameter.get_value()
        step = 1e-6 * jnp.abs(value)
        parameter.set_value(value + step)
        forward = goat.get_value(times=times)
        parameter.set_value(value - step)
        backward = goat.get_value(times=times)
        parameter.set_value(value)
        finite_differences.append((forward - backward) / (2 * step))

    testing.assert_allclose(np.array(gradient).flatten(), np.array(finite_differences).flatten(), rtol=1e-2)
