"""Test composite optimization."""

import numpy as np
import pytest

from paraqeet import UnitaryFidelity
from paraqeet.eom.schroedinger_equation import SchroedingerEquation
from paraqeet.hamiltonian.composite_hamiltonian import CompositeHamiltonian
from paraqeet.hamiltonian.coupling import Coupling
from paraqeet.hamiltonian.drive import Drive
from paraqeet.hamiltonian.transmon import TransmonHamiltonian
from paraqeet.optimization_map import OptimizationMap
from paraqeet.optimizers.scipy_optimizer import ScipyOptimizer, ScipyOptimizerGradient
from paraqeet.propagation.expm import Expm
from paraqeet.propagation.goat import GOAT
from paraqeet.quantity import Quantity
from paraqeet.signal.envelopes import FlatTopGaussianEnvelope
from paraqeet.signal.iq_mixer import IQMixer

FREQ1 = 5.5e9 * 2 * np.pi
ANHARM1 = -240e6 * 2 * np.pi

FREQ2 = 6.0e9 * 2 * np.pi
ANHARM2 = -200e6 * 2 * np.pi

COUPLINGSTR = 25e6 * 2 * np.pi

T_FINAL = 150e-9
RES = 100e9


@pytest.fixture
def tone():
    """Create a signal tone."""

    def _method(amp, t_final):
        tone = FlatTopGaussianEnvelope(
            amplitude=Quantity(
                amp * 2 * np.pi,
                min_value=0.8 * amp * 2 * np.pi,
                max_value=1.2 * amp * 2 * np.pi,
                unit="Hz",
            ),
            t_final=Quantity(
                t_final,
                min_value=0.8 * t_final,
                max_value=1.2 * t_final,
                unit="Hz",
            ),
        )
        return tone

    return _method


@pytest.fixture
def coupled_transmons(tone):
    """Create a coupled Transmon system."""
    freq1 = 6.0e9
    freq2 = 6.0002e9

    tone1 = tone(191e6, T_FINAL)
    tone2 = tone(9.18e6, T_FINAL)

    generator1 = IQMixer(
        envelopes=[tone1],
        frequency=Quantity(
            freq1 * 2 * np.pi,
            min_value=np.array(0.8 * freq1 * 2 * np.pi),
            max_value=np.array(1.2 * freq1 * 2 * np.pi),
            unit="Hz",
        ),
        phase=Quantity(
            0.01,
            min_value=np.array(-np.pi),
            max_value=np.array(np.pi),
            unit="rad",
        ),
    )

    generator2 = IQMixer(
        envelopes=[tone2],
        frequency=Quantity(
            freq2 * 2 * np.pi,
            min_value=np.array(0.8 * freq2 * 2 * np.pi),
            max_value=np.array(1.2 * freq2 * 2 * np.pi),
            unit="Hz",
        ),
        phase=Quantity(
            -0.39720756,
            min_value=np.array(-np.pi),
            max_value=np.array(np.pi),
            unit="rad",
        ),
    )

    transmon_hamiltonian_1 = TransmonHamiltonian(
        num_levels=3,
        frequency=Quantity(FREQ1, np.array(0.8 * FREQ1), np.array(1.2 * FREQ1), "Hz"),
        anharmonicity=Quantity(ANHARM1, np.array(1.2 * ANHARM1), np.array(0.8 * ANHARM1), "Hz"),
        drives=[],
    )

    drive_op1 = transmon_hamiltonian_1.annihilation_op + (transmon_hamiltonian_1.annihilation_op).conj().T
    drive1 = Drive(drive_op1, generator1)
    transmon_hamiltonian_1.drives = [drive1]

    transmon_hamiltonian_2 = TransmonHamiltonian(
        num_levels=3,
        frequency=Quantity(FREQ2, np.array(0.8 * FREQ2), np.array(1.2 * FREQ2), "Hz"),
        anharmonicity=Quantity(ANHARM2, np.array(1.2 * ANHARM2), np.array(0.8 * ANHARM2), "Hz"),
        drives=[],
    )

    drive_op2 = transmon_hamiltonian_2.annihilation_op + (transmon_hamiltonian_2.annihilation_op).conj().T
    drive2 = Drive(drive_op2, generator2)
    transmon_hamiltonian_2.drives = [drive2]

    coupling_op = np.kron(
        transmon_hamiltonian_1.annihilation_op + transmon_hamiltonian_1.annihilation_op.conj().T,
        transmon_hamiltonian_2.annihilation_op + transmon_hamiltonian_2.annihilation_op.conj().T,
    )

    coupling = Coupling(
        coupling_op,
        g_abs=Quantity(
            COUPLINGSTR,
            np.array(0.8 * COUPLINGSTR),
            np.array(1.2 * COUPLINGSTR),
            "Hz",
        ),
    )
    hamiltonian = CompositeHamiltonian([transmon_hamiltonian_1, transmon_hamiltonian_2], [coupling])
    model = SchroedingerEquation(
        hamiltonian_func=hamiltonian.get_value, hamiltonian_gradient_func=hamiltonian.get_gradient
    )
    prop = GOAT(
        Expm(
            eom_func=model.get_value,
            resolution=100e9,
            initial_state=np.identity(transmon_hamiltonian_1.dimension() * transmon_hamiltonian_2.dimension()),
        ),
        eom_gradient_func=model.get_gradient,
    )

    pauli_x = np.array([[0.0, 1], [1, 0.0]])
    pauli_z = np.array([[1, 0], [0.0, -1]])
    pauli_zx = np.exp(1j * np.pi / 4) * np.kron(pauli_z, pauli_x)
    cr_gate = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0], [0, 0, 1.0, 0]])

    cr_gate = pauli_zx @ cr_gate
    gate_fid = UnitaryFidelity(
        propagation_func=prop.get_value,
        propagation_and_gradient_func=prop.get_value_and_gradient,
        gate=cr_gate,
    )
    tone1_amp = tone1.get_parameters()[0]

    optmap = OptimizationMap()
    optmap.add(tone1, [tone1_amp])
    return gate_fid, optmap


@pytest.fixture
def opt(coupled_transmons):
    """Return Scipy optimizer from coupled transmons."""
    measure, optmap = coupled_transmons
    opt = ScipyOptimizer(measure_func=measure.calculate_normalized_scalar, optimization_map=optmap)
    opt.set_options({"maxiter": 5})
    return opt


@pytest.fixture
def grad_opt(coupled_transmons):
    """Return Scipy optimizer gradient."""
    measure, optmap = coupled_transmons
    opt = ScipyOptimizerGradient(
        measure_func=measure.get_value,
        measure_and_gradient_func=measure.get_value_and_gradient,
        optimization_map=optmap,
    )
    opt.set_options({"maxiter": 2})
    return opt


@pytest.mark.skip(reason="Projection needed")
def test_optim_finite_diff(opt):
    """Test optimization via finite differences."""
    # TODO: Add projection before testing.
    res = opt.optimize(times=T_FINAL)
    assert res.value < 0.1


@pytest.mark.skip(reason="Projection needed")
def test_optim_goat(grad_opt):
    """Test GOAT optimization."""
    # TODO: Add projection before testing.
    res = grad_opt.optimize(times=T_FINAL)
    assert res.value < 0.1
