"""Testing the GOAT optimization model."""

import numpy as np
import pytest

from paraqeet import UnitaryFidelity
from paraqeet.eom.master_equation import MasterEquation
from paraqeet.eom.schroedinger_equation import SchroedingerEquation
from paraqeet.hamiltonian.drive import Drive
from paraqeet.hamiltonian.qubit import Qubit, QubitHamiltonian
from paraqeet.measurement.fidelity import Fidelity
from paraqeet.measurement.utils import overlap_state_vector, overlap_vectorized_density_matrix, state_fidelity
from paraqeet.optimization_map import OptimizationMap
from paraqeet.optimizers.scipy_optimizer import ScipyOptimizer
from paraqeet.optimizers.scipy_optimizer_gradient import ScipyOptimizerGradient
from paraqeet.propagation.expm import Expm
from paraqeet.propagation.goat import GOAT
from paraqeet.propagation.utils import convert_dm_to_vec
from paraqeet.quantity import Quantity
from paraqeet.signal.envelopes import FlatTopGaussianEnvelope
from paraqeet.signal.iq_mixer import IQMixer

FREQ = 4.327884e9 * 2 * np.pi
T_FINAL = 13e-9

RES = 100e9
T1 = Quantity(10e-6, 1e-6, 100e-6)
TEMP = Quantity(10e-3, 1e-3, 50e-3)
T2STAR = Quantity(20e-6, 1e-6, 100e-6)


@pytest.fixture
def tone():
    """Return a cosine tone with a fixed error-function shaped envelope."""
    env = FlatTopGaussianEnvelope()
    env._t_up.set_value(T_FINAL / 5)
    env._t_down.set_value(4 * T_FINAL / 5)
    env._ramp_time.set_value(T_FINAL / 10)
    return env


@pytest.fixture
def gen(tone):
    """Generate a cosine tone."""
    gen = IQMixer(envelopes=[tone])
    return gen


@pytest.fixture(params=["OpenSystem", "ClosedSystem"])
def mode(request):
    return request.param


@pytest.fixture
def propagation(gen, mode):
    """Solve the equation of motion.

    By piecewise exponentiation with the scipy package.

    """
    init = np.array([[1.0], [0.0]])
    if mode == "OpenSystem":
        init = np.matmul(init, init.T)
        init = convert_dm_to_vec(init)

    qubit_hamiltonian = QubitHamiltonian(Quantity(FREQ, FREQ / 4, FREQ), drives=[])
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    drive = Drive(pauli_x, gen)
    qubit_hamiltonian.drives = [drive]
    open_qubit = Qubit(hamiltonian=qubit_hamiltonian, t1=T1, temp=TEMP, t2star=T2STAR)
    if mode == "OpenSystem":
        model = MasterEquation(
            hamiltonian_func=qubit_hamiltonian.get_value,
            hamiltonian_gradient_func=qubit_hamiltonian.get_gradient,
            jump_operators=open_qubit.get_jump_operators(),
        )
    elif mode == "ClosedSystem":
        model = SchroedingerEquation(
            hamiltonian_func=qubit_hamiltonian.get_value,
            hamiltonian_gradient_func=qubit_hamiltonian.get_gradient,
        )
    return Expm(eom_func=model.get_value, resolution=RES, initial_state=init), model


@pytest.fixture
def prop(propagation):
    """Add gradients to the propagation with GOAT."""
    expm, model = propagation
    return GOAT(expm, eom_gradient_func=model.get_gradient)


@pytest.fixture
def states(prop, mode):
    """Compare the overlap of the initial and final state."""
    init = np.array([[1.0], [0.0]])
    target = np.array([[0.0], [1.0]])
    overlap_func = overlap_state_vector
    fid_func = state_fidelity

    if mode == "OpenSystem":
        init = np.matmul(init, init.T)
        target = np.matmul(target, target.T)

        init = convert_dm_to_vec(init)
        target = convert_dm_to_vec(target)

        overlap_func = overlap_vectorized_density_matrix

    return Fidelity(
        propagation_func=prop.get_value,
        propagation_and_gradient_func=prop.get_value_and_gradient,
        target_states=target,
        overlap=overlap_func,
        fid=fid_func,
    )


@pytest.fixture
def gates(propagation, prop, mode):
    """Compare the propagator with a gate via the L2 norm."""
    if mode == "OpenSystem":
        pytest.skip("Gate optimization is only implemented for closed system.")
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    expm, _ = propagation
    expm.initial_state = np.identity(2)
    return UnitaryFidelity(
        propagation_func=prop.get_value,
        propagation_and_gradient_func=prop.get_value_and_gradient,
        gate=pauli_x,
    )


@pytest.fixture
def opt_map(gen):
    """Create an optimization map."""
    params = gen.get_parameters()
    params[0].set_value(0.5 * np.pi / T_FINAL)
    params[-2].set_value(1.01 * FREQ)
    optmap = OptimizationMap()
    # Not optimizing t_up, t_down, ramp_time
    optmap.add(gen, [params[0], params[-2], params[-1]])
    return optmap


@pytest.fixture
def grad_opt(states, opt_map):
    """Create a scipy optimizer gradient object over states."""
    return ScipyOptimizerGradient(measure_and_gradient_func=states.get_value_and_gradient, optimization_map=opt_map)


@pytest.fixture
def grad_gates_opt(gates, opt_map):
    """Create a scipy optimizer gradient object over gates."""
    return ScipyOptimizerGradient(measure_and_gradient_func=gates.get_value_and_gradient, optimization_map=opt_map)


@pytest.fixture
def opt(states, opt_map):
    """Create a scipy optimizer object over states."""
    return ScipyOptimizer(measure_func=states.calculate_normalized_scalar, optimization_map=opt_map)


@pytest.fixture
def gates_opt(gates, opt_map):
    """Create a scipy optimizer gradient object over gates."""
    return ScipyOptimizer(measure_func=gates.calculate_normalized_scalar, optimization_map=opt_map)


def test_optim_finite_diff(opt) -> None:
    """Check that the optimization goes below threshold."""
    res = opt.optimize(times=T_FINAL)
    assert res.value < 1e-2


def test_optim_goat(grad_opt) -> None:
    """Check that the optimization goes below threshold."""
    res = grad_opt.optimize(times=T_FINAL)
    assert res.value < 1e-2


def test_optim_gates_finite_diff(gates_opt) -> None:
    """Check that the optimization goes below threshold."""
    res = gates_opt.optimize(times=T_FINAL)
    assert res.value < 1e-2


def test_optim_goat_gates(grad_gates_opt) -> None:
    """Check that the optimization goes below threshold."""
    res = grad_gates_opt.optimize(times=T_FINAL)
    assert res.value < 1e-2
