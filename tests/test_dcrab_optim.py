"""Test dCRAB optimization using GOAT over GRAPE. This is same as the example 02E_GOAToverGRAPE_dCRAB"""

import numpy as np
import pytest

from paraqeet.eom.schroedinger_equation import SchroedingerEquation
from paraqeet.hamiltonian.drive import Drive
from paraqeet.hamiltonian.qubit import QubitHamiltonian
from paraqeet.measurement.fidelity import FidelityGRAPE
from paraqeet.measurement.goat_over_grape import GOATOverGRAPE
from paraqeet.measurement.utils import overlap_state_vector, state_fidelity
from paraqeet.optimization_map import OptimizationMap
from paraqeet.optimizers.dcrab_optimizer_gradient import DCRABOptimizerGradient
from paraqeet.propagation.expm import Expm
from paraqeet.propagation.grape import GRAPE
from paraqeet.propagation.utils import grape_operator_sandwich_function_closed
from paraqeet.quantity import Quantity
from paraqeet.signal.envelopes import DCRABEnvelope
from paraqeet.signal.pwc_generator import PWCGenerator
from paraqeet.signal.signal import FlatTopGaussianFilter

T_FINAL = 20e-9
TLIST = np.linspace(0, T_FINAL, 40)
EPS = 5 * np.pi * 1.0  # initial amplitude of the resonator (in MHz)
EPS_MAX = 10 * EPS  # maximum amplitude of the resonator (in MHz)


@pytest.fixture
def tone():
    tone = DCRABEnvelope(
        num_components=2,
        max_frequency=5 * 2 * np.pi,
        amplitude=Quantity(EPS * 1e6, -EPS_MAX * 1e6, EPS_MAX * 1e6, name="Amplitude"),
        t_final=Quantity(T_FINAL, 1 / 2 * T_FINAL, 2 * T_FINAL, name="t_final"),
        seed=18537,
    )
    return tone


@pytest.fixture
def gen(tone):
    smooth_tone = FlatTopGaussianFilter(
        envelopes=tone, t_final=Quantity(T_FINAL, 1 / 2 * T_FINAL, 2 * T_FINAL, name="t_final")
    )
    gen = PWCGenerator(envelopes=[smooth_tone], tlist=TLIST, max_amplitude=np.sqrt(2) * EPS_MAX * 1e6)
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
    init = np.array([[1.0], [0]])  # |0>
    return Expm(eom_func=model.get_value, resolution=1e9, initial_state=init)


@pytest.fixture
def prop(model, propagation):
    target = np.array([[0.0], [1]])  # |1>

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
    target = np.array([[0.0], [1]])  # |1>

    zeroone = FidelityGRAPE(
        propagation_func=prop.get_value,
        propagation_and_gradient_func=prop.get_value_and_gradient,
        target_states=target,
        overlap=overlap_state_vector,
        fid=state_fidelity,
    )
    return zeroone


@pytest.fixture
def opt_grad(tone, fid, gen, propagation):
    optmap = OptimizationMap()
    params = tone.get_parameters()
    optmap.add(tone, [params[0]] + params[2:])
    optmap.register_params_with_optimizables()

    goat = GOATOverGRAPE(fid, generators=[gen], propagation_resolution=propagation.resolution)
    opt_grad = DCRABOptimizerGradient(
        measure_func=goat.get_value,
        measure_and_gradient_func=goat.get_value_and_gradient,
        optimization_map=optmap,
        super_iteration_every=150,
        max_super_iteration_num=5,
        print_every_iteration_num=10,
        super_iteration_tol=1e-9,
        seed=19573,
    )
    return opt_grad


def test_goat_over_grape(opt_grad):
    res = opt_grad.optimize(TLIST)
    assert res.value < 1e-4
