"""Testing the GRAPE optimization of a TLS system."""

import numpy as np
import pytest

from paraqeet.eom.master_equation import MasterEquation
from paraqeet.eom.schroedinger_equation import SchroedingerEquation
from paraqeet.hamiltonian.drive import Drive
from paraqeet.hamiltonian.qubit import Qubit, QubitHamiltonian
from paraqeet.measurement.fidelity import (
    FidelityGRAPE,
)
from paraqeet.measurement.utils import (
    overlap_density_matrix,
    overlap_state_vector,
    overlap_vectorized_density_matrix,
    state_fidelity,
)
from paraqeet.optimization_map import OptimizationMap
from paraqeet.optimizers.scipy_optimizer import ScipyOptimizerGradient
from paraqeet.propagation.expm import Expm
from paraqeet.propagation.grape import GRAPE
from paraqeet.propagation.utils import (
    convert_dm_to_vec,
    grape_operator_sandwich_function_closed,
    grape_operator_sandwich_function_open,
    lindblad_step,
    reverse_lindblad_step,
    schrodinger_step,
)
from paraqeet.propagation.vern7 import Vern7
from paraqeet.quantity import Quantity
from paraqeet.signal.envelopes import GaussEnvelope
from paraqeet.signal.pwc_generator import PWCGenerator

T_FINAL = 20e-9
FREQ = 1e6
TLIST = np.linspace(0, T_FINAL, 21)
DELTAT = TLIST[1] - TLIST[0]
T1 = Quantity(10e-6, 1e-9, 100e-6)
TEMP = Quantity(10e-3, 1e-3, 50e-3)
T2STAR = Quantity(10e-6, 1e-9, 100e-6)


@pytest.fixture
def tone():
    tone = GaussEnvelope(amplitude=Quantity(np.pi / T_FINAL / 3, -np.pi / T_FINAL, np.pi / T_FINAL))
    tone.t_final.set_value(T_FINAL)
    return tone


@pytest.fixture
def pwc_gen(tone):
    gen = PWCGenerator(envelopes=[tone], tlist=TLIST)
    gen.multiply_flat_top = True
    gen.max_amplitude = 2 * 1e8
    return gen


@pytest.fixture(params=["OpenSystem", "ClosedSystem"])
def mode(request):
    return request.param


@pytest.fixture(params=["expm", "ode"])
def solver(request):
    return request.param


@pytest.fixture
def model(pwc_gen, mode):
    init = np.array([[1.0], [0.0]])
    if mode == "OpenSystem":
        init = np.matmul(init, init.T)
        init = convert_dm_to_vec(init)

    qubit_hamiltonian = QubitHamiltonian(Quantity(FREQ, FREQ / 4, FREQ), drives=[])
    drive = Drive(qubit_hamiltonian.sigma_minus, pwc_gen, add_hermitian=True)
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
    return model


@pytest.fixture
def states(model, mode, solver):
    """Compare the overlap of the initial and final state."""
    init = np.array([[1.0], [0.0]])
    target = np.array([[0.0], [1]])
    overlap_func = overlap_state_vector
    operator_sandwich_func = grape_operator_sandwich_function_closed

    if mode == "OpenSystem" and solver == "expm":
        init = np.matmul(init, init.T)
        target = np.matmul(target, target.T)
        init = convert_dm_to_vec(init)
        target = convert_dm_to_vec(target)
        overlap_func = overlap_vectorized_density_matrix
    elif mode == "OpenSystem" and solver == "ode":
        init = np.matmul(init, init.T)
        target = np.matmul(target, target.T)
        overlap_func = overlap_density_matrix
        operator_sandwich_func = grape_operator_sandwich_function_open

    # The EOM of an open system is a superoperator acting on the vectorized density matrix for
    # ``Expm``, which propagates by matrix multiplication, and the Hamiltonian plus a list of
    # collapse operators for the ODE solver, which builds the dissipator in its step function.
    if solver == "expm":
        propagation = Expm(eom_func=model.get_value, resolution=1 / DELTAT, initial_state=init)
        eom_gradient_func = model.get_gradient
        reverse_step_func = None

    elif mode == "ClosedSystem" and solver == "ode":
        propagation = Vern7(
            eom_func=model.get_value,
            resolution=10e9,
            initial_state=init,
            step_function=schrodinger_step,
        )
        eom_gradient_func = model.get_gradient
        # Daggering the EOM is all the backward propagation of a closed system needs.
        reverse_step_func = None

    elif mode == "OpenSystem" and solver == "ode":
        propagation = Vern7(
            eom_func=model.get_eom_ode_propagation,
            resolution=10e9,
            initial_state=init,
            step_function=lindblad_step,
            jump_operators=model.jump_operators,
        )
        eom_gradient_func = model.get_eom_gradient_ode_propagation
        # The dissipator is not part of the EOM here, so the backward propagation needs its adjoint.
        reverse_step_func = reverse_lindblad_step

    prop_method = GRAPE(
        propagation,
        eom_gradient_func=eom_gradient_func,
        target_state=target,
        operator_sandwich_function=operator_sandwich_func,
        reverse_step_function=reverse_step_func,
    )

    return FidelityGRAPE(
        propagation_func=prop_method.get_value,
        propagation_and_gradient_func=prop_method.get_value_and_gradient,
        target_states=target,
        overlap=overlap_func,
        fid=state_fidelity,
    )


@pytest.fixture
def opt_map(pwc_gen):
    """Create an optimization map."""
    optmap = OptimizationMap()
    optmap.add(pwc_gen)
    return optmap


@pytest.fixture
def opt(states, opt_map):
    """Create a scipy optimizer gradient object over states."""
    return ScipyOptimizerGradient(
        measure_func=states.get_value, measure_and_gradient_func=states.get_value_and_gradient, optimization_map=opt_map
    )


def test_optim_grape(opt) -> None:
    """Check that the optimization goes below threshold."""
    res = opt.optimize(times=TLIST)
    assert res.value < 1e-2
