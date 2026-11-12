Arbitrary bosonic state preparation using GRAPE
===============================================

In this notebook, we implement a standard application of GRAPE, namely
the preparation of an arbitrary state of a bosonic mode, such as a
resonant mode of a microwave cavity. In the notebook, we use GRAPE
(Khaneja et al., 2005) :cite:p:`khaneja2005optimal` and mostly follow
the optimization strategy of (Heeres et al., 2017)
:cite:p:`heeres2017implementing`. The parameters are taken from
(Eickbusch et al., 2022) :cite:p:`eickbusch2022fast` Table S1, without
considering the anharmonicity for simplicity and with the exception of
the dispersive shift taken to be :math:`10` times larger for simulation
purposes. Our starting point is the well-known Jaynes-Cummings
Hamiltonian in the dispersive regime, which describes an interaction of
a qubit with a single bosonic mode when the characteristic qubit
frequency :math:`\omega_{q}` is far detuned from the one of the
resonator :math:`\omega_{r}` (see for instance (Blais et al., 2021)
:cite:p:`blais2021circuit` for more details)

.. math::


   H_{\mathrm{disp}} = \omega_{r} a^{\dagger} a + \frac{\omega_{q}}{2} \sigma_z + \chi \sigma_z a^{\dagger} a,

where we set :math:`\hbar = 1`. The parameter :math:`\chi` is the
dispersive shift, which represents a qubit-state-dependent shift of the
frequency of the resonator. Note that we set conventionally
:math:`\sigma_z = | e \rangle \langle e | - |g \rangle \langle g |`,
with :math:`|e \rangle, |g \rangle` the excited and ground state of the
qubit, respectively. We additionally introduce a drive Hamiltonian on
both the qubit and the resonator mode, which we write directly within
the rotating wave approximation

.. math::


   H_{d, r}(t) = \varepsilon_{r}(t) e^{i \omega_{d, r} t} a + \varepsilon_{r}^*(t) e^{-i \omega_{d, r} t} a^{\dagger} , \quad H_{d, q}(t) = \varepsilon_{q}(t) e^{i \omega_{d, q} t} \sigma_- + \varepsilon_{q}^*(t) e^{-i \omega_{d, q} t} \sigma_+.

The total Hamiltonian is

.. math::


   H(t) = H_{\mathrm{disp}} + H_{d, r}(t) + H_{d, q}(t).

We assume that the qubit and the resonator are both driven resonantly,
i.e., :math:`\omega_{d, q} = \omega_q` and
:math:`\omega_{d,r}=\omega_r`, and work in the interaction picture
(rotating frame) with reference Hamiltonian

.. math::


   H_{\mathrm{ref}} = \omega_{r} a^{\dagger} a + \frac{\omega_{q}}{2} \sigma_z,

so that the Hamiltonian in the interaction picture is

.. math::


   H_{I}(t) = e^{i H_{\mathrm{ref}} t} (H(t) - H_{\mathrm{ref}}) e^{-i H_{\mathrm{ref}} t} = \chi \sigma_z a^{\dagger} a + \varepsilon_{r}(t) a + \varepsilon_{r}^*(t) a^{\dagger}  + \varepsilon_{q}(t) \sigma_- + \varepsilon_{q}^*(t) \sigma_+,

Our goal is to prepare the resonator in a Fock state
:math:`| n \rangle, \, n \in \mathbb{N}`, while keeping the qubit in the
ground state :math:`| g \rangle`. Thus, the target state is

.. math::


   | \Psi_{\mathrm{target}} \rangle = |n, g \rangle.

In the notebook, :math:`n` is set to :math:`2` as an example. We assume
that the system starts in the initial state

.. math::


   | \Psi_{\mathrm{initial}} \rangle = |0, g \rangle.

Let :math:`\varepsilon(t) = (\varepsilon_{r}(t), \varepsilon_{q}(t))`.
For a fixed time :math:`T`, the system evolves to a state
:math:`| \Psi(T; \varepsilon(t)) \rangle = U(T; \varepsilon(t))  | \Psi_{\mathrm{initial}} \rangle`.
We thus want to maximize the state fidelity, i.e., the overlap between
:math:`| \Psi_{\mathrm{target}} \rangle` and :math:`| \Psi(t) \rangle`:

.. math::


   \mathcal{F}(\varepsilon(t))  = | \langle \Psi(T; \varepsilon(t) ) )| \Psi_{\mathrm{target}} \rangle|^2.

In GRAPE we consider piecewise constant pulses with time step
:math:`\Delta t`. Each constant can take values in a certain subset
:math:`\mathcal{I} \in \mathbb{C}`. We denote by :math:`R` the range of
the piecewise constant pulse, i.e.,
:math:`R = \mathrm{max}_{\varepsilon, \varepsilon' \in \mathcal{I}} | \varepsilon - \varepsilon'|`.
Thus, in what follows :math:`R_r` and :math:`R_q` represent the range of
the resonator and qubit pulses, respectively.

.. code:: ipython3

    import matplotlib.pyplot as plt
    import numpy as np
    
    from paraqeet.eom.schroedinger_equation import SchroedingerEquation
    from paraqeet.hamiltonian.composite_hamiltonian import CompositeHamiltonian
    from paraqeet.hamiltonian.coupling import Coupling
    from paraqeet.hamiltonian.drive import Drive
    from paraqeet.hamiltonian.qubit import QubitHamiltonian
    from paraqeet.hamiltonian.resonator import ResonatorHamiltonian
    from paraqeet.measurement.fidelity import FidelityGRAPE
    from paraqeet.measurement.smoothness import Smoothness
    from paraqeet.measurement.weighted_sum_goal import WeightedSumGoal
    from paraqeet.optimization_map import OptimizationMap
    from paraqeet.optimizers.scipy_optimizer import ScipyOptimizerGradient
    from paraqeet.propagation import GRAPE, Expm
    from paraqeet.quantity import Quantity
    from paraqeet.signal.envelopes import GaussEnvelope
    from paraqeet.signal.pwc_generator import PWCGenerator

We initialize our pulses as simple Gaussian envelopes. Additionally, we
require fixed ranges for the minimum and maximum amplitudes.

.. code:: ipython3

    # in seconds (See Eickbusch et al https://arxiv.org/abs/2111.06414 S9 A)
    delta_sampling = 33e-9
    n_pwc = 40  # number of piecewise constants in the pulse
    t_final = n_pwc * delta_sampling  # for now just set up for trying
    tlist = np.linspace(0, t_final, n_pwc + 1)
    eps_res = 2 * np.pi * 1.0  # initial amplitude of the resonator (in MHz)
    eps_max_res = 5 * eps_res  # maximum amplitude of the resonator (in MHz)
    eps_qubit = 2 * np.pi * 1.0  # initial amplitude of the qubit (in MHz)
    eps_max_qubit = 5 * eps_qubit  # maximum amplitude of the resonator (in MHz)
    tone_res = GaussEnvelope(
        amplitude=Quantity(eps_res * 1e6, -eps_max_res * 1e6, eps_max_res * 1e6),
        t_final=Quantity(t_final, 1 / 2 * t_final, 2 * t_final),
    )
    tone_qubit = GaussEnvelope(
        amplitude=Quantity(eps_qubit * 1e6, -eps_max_qubit * 1e6, eps_max_qubit * 1e6),
        t_final=Quantity(t_final, 1 / 2 * t_final, 2 * t_final),
    )
    gen_res = PWCGenerator(envelopes=[tone_res], tlist=tlist)
    gen_res.multiply_flat_top = True
    
    gen_qubit = PWCGenerator(envelopes=[tone_qubit], tlist=tlist)
    gen_qubit.multiply_flat_top = True

We plot the resonator tone and compare it to its smooth version.
Similarly one can plot the qubit tone.

.. code:: ipython3

    from plotting import plot_signal
    
    ts = np.linspace(0, t_final, 1001)
    fig, ax = plt.subplots(1, figsize=(5, 3))
    
    plot_signal(gen_res, ts, ax, linestyle="--", label="Resonator pulse")
    plot_signal(gen_qubit, ts, ax, linestyle="-", label="Qubit pulse")
    plt.legend()
    plt.show()



.. image:: 08B_Bosonic_grape_state_preparation_files/08B_Bosonic_grape_state_preparation_5_0.png


We define qubit and resonator parameters, as well as the list of
different Fock truncation numbers. We also set the target Fock state.

For now we set the Fock-state truncation at small values (3 and 4) for
testing the method. Later we demonstrate the method with (30 and 31)
levels in the resonator.

.. code:: ipython3

    # Increasing the number of Fock states gives a more realistic scenario, at the price of increasing the simulation time
    n_fock_truncation_list = [3, 4]
    fock_target = 2
    n_times = 1001
    
    omega_range_factor = 1e-3  # just for initialization of the Quantity object
    omega_res = 2 * np.pi * 5.26e9
    omega_qubit = 2 * np.pi * 6.65e9
    
    omega_drive_res = omega_res
    omega_drive_qubit = omega_qubit
    
    detuning_drive_res = omega_res - omega_drive_res
    detuning_drive_qubit = omega_qubit - omega_drive_qubit

Due to the infinite-dimensional nature of the Hilbert space of the
resonator, it is necessary to introduce a Fock state truncation number
:math:`N_{\mathrm{T}}`. This creates the problem that, given certain
pulses, :math:`\mathcal{F}` depends on the choice of
:math:`N_{\mathrm{T}}`. Following (Heeres et al., 2017)
:cite:p:`heeres2017implementing`, we thus consider
:math:`N_{\mathrm{T}} \in \{N_{\mathrm{T}}^{(\mathrm{min})}, N_{\mathrm{T}}^{(\mathrm{min})} + 1, \dots, N_{\mathrm{T}}^{(\mathrm{max})} \}`,
and introduce a penalty when having different values of fidelities for
different truncation numbers. Thus, we create different systems, and
accordingly fidelity measures, for the different Fock truncation
numbers.

.. code:: ipython3

    from paraqeet.measurement.utils import overlap_state_vector, state_fidelity
    from paraqeet.propagation.utils import grape_operator_sandwich_function_closed
    
    
    def create_experiment(n_fock_truncation_list, fock_target):
        """
        Create the initial and target states, for the given Fock numbers.
        Also create the propagations and measurements for the given truncation numbers and target Fock states.
        """
        resonator_list = []
        coupling_list = []
        model_list = []
        hamiltonian_list = []
        prop_list = []
        initial_state_list = []
        target_state_list = []
        fock_number_op_list = []
        fid_list = []
    
        qubit_hamiltonian = QubitHamiltonian(
            frequency=Quantity(
                value=detuning_drive_qubit,
                min_value=-omega_range_factor * omega_qubit,
                max_value=omega_range_factor * omega_qubit,
            ),
            drives=[],
        )
    
        drive_qubit = Drive(qubit_hamiltonian.sigma_minus, gen_qubit, add_hermitian=True)
        qubit_hamiltonian.drives = [drive_qubit]
    
        ground_state_qubit = np.array([[0.0], [1.0]])
    
        # We take it to be 10 times the value in Eickbusch et al to speed up the simulation
        chi = 2 * np.pi * 32.81e3 * 10
    
        for n in n_fock_truncation_list:
            resonator_hamiltonian = ResonatorHamiltonian(
                frequency=Quantity(
                    value=detuning_drive_res,
                    min_value=-omega_range_factor * omega_res,
                    max_value=omega_range_factor * omega_res,
                ),
                num_fock=n,
                drives=[],
            )
    
            drive_res = Drive(resonator_hamiltonian.annihilation_op, gen_res, add_hermitian=True)
            resonator_hamiltonian.drives = [drive_res]
    
            resonator_list.append(resonator_hamiltonian)
    
            coupling_op = np.kron(resonator_hamiltonian.num_op, np.array([[1.0, 0.0], [0.0, -1.0]]))
            coupling = Coupling(
                coupling_op,
                g_abs=Quantity(value=chi, min_value=chi / 2, max_value=2 * chi),
            )
            coupling_list.append(coupling)
            ham = CompositeHamiltonian([resonator_hamiltonian, qubit_hamiltonian], [coupling])
            hamiltonian_list.append(ham)
            model = SchroedingerEquation(hamiltonian_func=ham.get_value, hamiltonian_gradient_func=ham.get_gradient)
            model_list.append(model)
    
            initial_res_state = np.zeros([n, 1], dtype=complex)
            initial_res_state[0, 0] = 1  # vacuum state
            initial_state = np.kron(initial_res_state, ground_state_qubit)
            initial_state_list.append(initial_state)
    
            target_res_state = np.zeros([n, 1], dtype=complex)
            target_res_state[fock_target, 0] = 1
            target_state = np.kron(target_res_state, ground_state_qubit)
            target_state_list.append(target_state)
    
            n_op = np.kron(np.diag([x for x in range(n)]), np.identity(2))
            fock_number_op_list.append(n_op)
    
            # Propagation dt has to be less than `delta_sampling = 33e-9`. Set dt = 30e-9.
            propagation = Expm(eom_func=model.get_value, resolution=1 / (30e-9), initial_state=initial_state)
            prop = GRAPE(
                propagation,
                eom_gradient_func=model.get_gradient,
                target_state=target_state,
                operator_sandwich_function=grape_operator_sandwich_function_closed,
                order=3,
            )
    
            prop_list.append(prop)
    
            fid = FidelityGRAPE(
                propagation_func=prop.get_value,
                propagation_and_gradient_func=prop.get_value_and_gradient,
                target_states=target_state,
                overlap=overlap_state_vector,
                fid=state_fidelity,
            )
    
            fid_list.append(fid)
    
        return initial_state_list, target_state_list, fock_number_op_list, prop_list, fid_list
    
    
    initial_state_list, target_state_list, fock_number_op_list, prop_list, meas_list = create_experiment(
        n_fock_truncation_list, fock_target
    )

Furthermore, following (Heeres et al., 2017)
:cite:p:`heeres2017implementing` we also introduce a penalty for
non-smooth pulses. In particular, we consider the (normalized) sum of
consecutive square differences of the pulse pixels as cost function (see
Eq. 21 in the supplementary material of (Heeres et al., 2017)
:cite:p:`heeres2017implementing`) for both the resonator and the qubit
pulses:

.. math:: g_{\mathrm{smooth}, r} (\varepsilon(t)) = 1.0 - \frac{1}{(N_{\mathrm{PWC}} - 1) R_r^2}\sum_{n=0}^{N_{\mathrm{PWC}} - 1} | \varepsilon_{r}((n+1) \Delta t) - \varepsilon_{r}((n) \Delta t) |^2,

.. math:: \quad g_{\mathrm{smooth}, q} (\varepsilon(t)) = 1.0- \frac{1}{(N_{\mathrm{PWC}} - 1) R_q^2} \sum_{n=0}^{N_{\mathrm{PWC}} - 1} | \varepsilon_{q}((n+1) \Delta t) - \varepsilon_{q}((n) \Delta t) |^2

.. code:: ipython3

    res_smoothness = Smoothness(pwc_generator=gen_res)
    qubit_smoothness = Smoothness(pwc_generator=gen_qubit)
    meas_list.append(res_smoothness)
    meas_list.append(qubit_smoothness)

We consider a cost function of the form

.. math::


   C(\varepsilon(t) ) = w_1 \sum_{N = N_{\mathrm{T}}^{(\mathrm{min})}}^{ N_{\mathrm{T}}^{(\mathrm{max})}} \mathcal{F}_{N} (\varepsilon(t) ) + w_2 g_{\mathrm{smooth}, r} (\varepsilon(t)) + w_3 g_{\mathrm{smooth}, q} (\varepsilon(t))   - \frac{w_4}{2} \sum_{N, N' = N_{\mathrm{T}}^{(\mathrm{min})}}^{ N_{\mathrm{T}}^{(\mathrm{max})}} \left[\mathcal{F}_{N}(\varepsilon(t))- \mathcal{F}_{N'} (\varepsilon(t) ) \right]^2,

that we want to maximize. This weighted cost function can be constructed
using the class WeightedSumGoal.

.. code:: ipython3

    weights = np.array([1.0 for _ in range(len(meas_list))])
    weights[-1] = 10.0
    weights[-2] = 10.0
    weight_sum_of_squares = -0.05
    total_sum_of_weights = np.sum(weights) + weight_sum_of_squares
    weights = weights / total_sum_of_weights
    weight_sum_of_squares = weight_sum_of_squares / total_sum_of_weights
    meas_list_bool = [True for meas in meas_list]
    meas_list_bool[-1] = False
    meas_list_bool[-2] = False
    sum_of_squares_options = {"weight": weight_sum_of_squares, "meas_bool": meas_list_bool}
    
    goal = WeightedSumGoal(meas_list, weights, sum_of_squares_options=sum_of_squares_options)

We can plot the initial pulses, as well as the corresponding relevant
dynamics.

.. code:: ipython3

    def plot_states_and_fock_number(item=0):
        """Plot the states."""
        ts = np.linspace(0, t_final, n_times)
        states = prop_list[item].get_value(ts)
        sig_res = gen_res.get_value(ts)
        sig_qubit = gen_qubit.get_value(ts)
        pop_initial_state = (np.abs(initial_state_list[item].conj().T @ states) ** 2).flatten()
        pop_target_state = (np.abs(target_state_list[item].conj().T @ states) ** 2).flatten()
    
        pop_initial_state = (np.abs(initial_state_list[item].conj().T @ states) ** 2).flatten()
    
        n_fock_avg = np.zeros(n_times, dtype=float)
        for k in range(n_times):
            n_fock_avg[k] = np.real((states[k].conj().T @ fock_number_op_list[item] @ states[k])[0, 0])
    
        _, ax = plt.subplots(4, figsize=(4, 10), sharex=True)
        ax[0].plot(ts / 1e-9, np.real(sig_res) / 1e6 / (2 * np.pi), label="I")
        ax[0].plot(ts / 1e-9, np.imag(sig_res) / 1e6 / (2 * np.pi), label="Q")
        ax[0].legend(loc=1)
        ax[0].set_ylabel(r"Field [MHz / $2\pi$]")
        ax[0].set_title("Resonator")
        ax[0].grid(True, linestyle=(1, (1, 5)), linewidth=1)
    
        ax[1].plot(ts / 1e-9, np.real(sig_qubit) / 1e6 / (2 * np.pi), label="I")
        ax[1].plot(ts / 1e-9, np.imag(sig_qubit) / 1e6 / (2 * np.pi), label="Q")
        ax[1].legend(loc=1)
        ax[1].set_ylabel(r"Field [MHz / $2\pi$]")
        ax[1].set_title("Qubit")
        ax[1].grid(True, linestyle=(1, (1, 5)), linewidth=1)
    
        ax[2].plot(ts / 1e-9, pop_initial_state, label="$|0,0 \\rangle$")
        ax[2].plot(ts / 1e-9, pop_target_state, label="$|2, 0 \\rangle$")
        ax[2].legend(loc="best")
        ax[2].set_ylabel("Population")
        ax[2].grid(True, linestyle=(1, (1, 5)), linewidth=1)
    
        ax[3].plot(ts / 1e-9, n_fock_avg)
        ax[3].set_ylabel("$\\langle n \\rangle$")
        if n_fock_truncation_list[item] < 10:
            y_fock_ticks = np.arange(n_fock_truncation_list[item])
        else:
            y_fock_ticks = np.arange(n_fock_truncation_list[item])[::5]
        ax[3].set_yticks(y_fock_ticks)
        ax[3].grid(axis="y", which="major")
        ax[3].minorticks_on()
        ax[3].grid(True, axis="x", linestyle=(1, (1, 5)), linewidth=1)
    
        ax[-1].set_xlabel("Time [ns]")
        plt.show()
    
    
    plot_states_and_fock_number()



.. image:: 08B_Bosonic_grape_state_preparation_files/08B_Bosonic_grape_state_preparation_15_0.png


We can compute the fidelities for the different truncation numbers

.. code:: ipython3

    print(f"Fidelity at N_T={n_fock_truncation_list[0]} = {meas_list[0].get_value(tlist)}")
    print(f"Fidelity at N_T={n_fock_truncation_list[1]} = {meas_list[1].get_value(tlist)}")


.. parsed-literal::

    Fidelity at N_T=3 = 0.14410356849601133


.. parsed-literal::

    Fidelity at N_T=4 = 0.06788860193208278


which are quite poor! We now proceed with the pulse optimization.

.. code:: ipython3

    max_iter = 500  # set to 1000 for a good result; set to 10 for a quick example
    optmap = OptimizationMap()
    optmap.add(gen_res, gen_res.get_parameters())
    optmap.add(gen_qubit, gen_qubit.get_parameters())
    opt = ScipyOptimizerGradient(measure_func=goal.get_value, measure_and_gradient_func=goal.get_value_and_gradient, optimization_map=optmap)
    opt.set_options({"maxfun": max_iter})
    optmap.register_params_with_optimizables()

.. code:: ipython3

    optmap




.. parsed-literal::

    ==== <class 'paraqeet.signal.pwc_generator.PWCGenerator'> ====
    [Inphase: [3.13e+03, 6.69e+03, 1.37e+04, 2.71e+04, 5.15e+04, 9.38e+04, 1.64e+05, 2.76e+05, 4.46e+05, 6.93e+05, 1.03e+06, 1.48e+06, 2.04e+06, 2.7e+06, 3.43e+06, 4.19e+06, 4.92e+06, 5.54e+06, 6.01e+06, 6.25e+06, 6.25e+06, 6.01e+06, 5.54e+06, 4.92e+06, 4.19e+06, 3.43e+06, 2.7e+06, 2.04e+06, 1.48e+06, 1.03e+06, 6.93e+05, 4.46e+05, 2.76e+05, 1.64e+05, 9.38e+04, 5.15e+04, 2.71e+04, 1.37e+04, 6.69e+03, 3.13e+03], out-of-phase: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    
    ==== <class 'paraqeet.signal.pwc_generator.PWCGenerator'> ====
    [Inphase: [3.13e+03, 6.69e+03, 1.37e+04, 2.71e+04, 5.15e+04, 9.38e+04, 1.64e+05, 2.76e+05, 4.46e+05, 6.93e+05, 1.03e+06, 1.48e+06, 2.04e+06, 2.7e+06, 3.43e+06, 4.19e+06, 4.92e+06, 5.54e+06, 6.01e+06, 6.25e+06, 6.25e+06, 6.01e+06, 5.54e+06, 4.92e+06, 4.19e+06, 3.43e+06, 2.7e+06, 2.04e+06, 1.48e+06, 1.03e+06, 6.93e+05, 4.46e+05, 2.76e+05, 1.64e+05, 9.38e+04, 5.15e+04, 2.71e+04, 1.37e+04, 6.69e+03, 3.13e+03], out-of-phase: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]




.. code:: ipython3

    %%time
    opt.optimize(gen_res.tlist)


.. parsed-literal::

    Iteration   10 | Infid = 2.419405e-02


.. parsed-literal::

    Iteration   20 | Infid = 1.911555e-02


.. parsed-literal::

    Iteration   30 | Infid = 1.391941e-02


.. parsed-literal::

    Iteration   40 | Infid = 1.237015e-02


.. parsed-literal::

    Iteration   50 | Infid = 1.103964e-02


.. parsed-literal::

    Iteration   60 | Infid = 1.007346e-02


.. parsed-literal::

    Iteration   70 | Infid = 9.334688e-03


.. parsed-literal::

    Iteration   80 | Infid = 9.141002e-03


.. parsed-literal::

    Iteration   90 | Infid = 9.105647e-03


.. parsed-literal::

    Iteration  100 | Infid = 9.078607e-03


.. parsed-literal::

    Iteration  110 | Infid = 9.022714e-03


.. parsed-literal::

    Iteration  120 | Infid = 8.817123e-03


.. parsed-literal::

    Iteration  130 | Infid = 8.682394e-03


.. parsed-literal::

    Iteration  140 | Infid = 8.679748e-03


.. parsed-literal::

    Iteration  150 | Infid = 8.674133e-03


.. parsed-literal::

    CPU times: user 2min 10s, sys: 1.93 s, total: 2min 11s
    Wall time: 29.9 s




.. parsed-literal::

    {'status': 1, 'value': 0.00867413182782295, 'iterations': 151, 'evaluations': 319, 'message': 'CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH'}



We can now plot the new pulses and the corresponding dynamics. If the
number of iterations (max_iter) is chosen to be large enough, we should
see an improvement.

.. code:: ipython3

    plot_states_and_fock_number()



.. image:: 08B_Bosonic_grape_state_preparation_files/08B_Bosonic_grape_state_preparation_23_0.png


The new fidelities are

.. code:: ipython3

    print(f"Fidelity at N_T={n_fock_truncation_list[0]} = {meas_list[0].get_value(tlist)}")
    print(f"Fidelity at N_T={n_fock_truncation_list[1]} = {meas_list[1].get_value(tlist)}")


.. parsed-literal::

    Fidelity at N_T=3 = 0.9218161114767965
    Fidelity at N_T=4 = 0.8736466275497098


Setting truncation to higher values
-----------------------------------

Here we increase the system truncation to 15 and 20 levels and rerun the
entire simulation. For more accurate results, we advise the reader to
increase the truncation to 30 and 31 levels.

*Note - The following takes about 5 minutes to run on an AMD-EPYC Milan
processor with 64 cores (might take more depending on the number of CPU
cores available).*

We first reset the generator parameters (by redefining them), and
redefine the resonator with higher truncation numbers.

.. code:: ipython3

    # in seconds (See Eickbusch et al https://arxiv.org/abs/2111.06414 S9 A)
    delta_sampling = 33e-9
    n_pwc = 40  # number of piecewise constants in the pulse
    t_final = n_pwc * delta_sampling  # for now just set up for trying
    tlist = np.linspace(0, t_final, n_pwc + 1)
    eps_res = 2 * np.pi * 1.0  # initial amplitude of the resonator (in MHz)
    eps_max_res = 5 * eps_res  # maximum amplitude of the resonator (in MHz)
    eps_qubit = 2 * np.pi * 1.0  # initial amplitude of the qubit (in MHz)
    eps_max_qubit = 5 * eps_qubit  # maximum amplitude of the resonator (in MHz)
    tone_res = GaussEnvelope(
        amplitude=Quantity(eps_res * 1e6, -eps_max_res * 1e6, eps_max_res * 1e6),
        t_final=Quantity(t_final, 1 / 2 * t_final, 2 * t_final),
    )
    tone_qubit = GaussEnvelope(
        amplitude=Quantity(eps_qubit * 1e6, -eps_max_qubit * 1e6, eps_max_qubit * 1e6),
        t_final=Quantity(t_final, 1 / 2 * t_final, 2 * t_final),
    )
    gen_res = PWCGenerator(envelopes=[tone_res], tlist=tlist)
    gen_qubit = PWCGenerator(envelopes=[tone_qubit], tlist=tlist)

.. code:: ipython3

    n_fock_truncation_list = [15, 20]  # Increase to 30, 31 for more accurate results
    fock_target = 2
    n_times = 1001
    
    omega_range_factor = 1e-3  # just for initialization of the Quantity object
    omega_res = 2 * np.pi * 5.26e9
    omega_qubit = 2 * np.pi * 6.65e9
    
    omega_drive_res = omega_res
    omega_drive_qubit = omega_qubit
    
    detuning_drive_res = omega_res - omega_drive_res
    detuning_drive_qubit = omega_qubit - omega_drive_qubit
    
    initial_state_list, target_state_list, fock_number_op_list, prop_list, meas_list = create_experiment(
        n_fock_truncation_list, fock_target
    )

Let’s also add the smoothness penalty to the measurement

.. code:: ipython3

    res_smoothness = Smoothness(pwc_generator=gen_res)
    qubit_smoothness = Smoothness(pwc_generator=gen_qubit)
    meas_list.append(res_smoothness)
    meas_list.append(qubit_smoothness)
    
    weights = np.array([1.0 for _ in range(len(meas_list))])
    weights[-1] = 10.0
    weights[-2] = 10.0
    weight_sum_of_squares = -0.05
    total_sum_of_weights = np.sum(weights) + weight_sum_of_squares
    weights = weights / total_sum_of_weights
    weight_sum_of_squares = weight_sum_of_squares / total_sum_of_weights
    meas_list_bool = [True for meas in meas_list]
    meas_list_bool[-1] = False
    meas_list_bool[-2] = False
    sum_of_squares_options = {"weight": weight_sum_of_squares, "meas_bool": meas_list_bool}
    
    goal = WeightedSumGoal(meas_list, weights, sum_of_squares_options=sum_of_squares_options)

And then plot the dynamics of the resonator under the unoptimized pulse

.. code:: ipython3

    plot_states_and_fock_number()



.. image:: 08B_Bosonic_grape_state_preparation_files/08B_Bosonic_grape_state_preparation_33_0.png


Initial fidelity before optimization

.. code:: ipython3

    print(f"Fidelity at N_T={n_fock_truncation_list[0]} = {meas_list[0].get_value(tlist)}")
    print(f"Fidelity at N_T={n_fock_truncation_list[1]} = {meas_list[1].get_value(tlist)}")


.. parsed-literal::

    Fidelity at N_T=15 = 0.014283975566438286


.. parsed-literal::

    Fidelity at N_T=20 = 0.014283963209907323


We redefine the optimizer and perform the optimization again with higher
truncation numbers

.. code:: ipython3

    max_iter = 200  # set to 1000 for a good result; set to 10 for a quick example
    optmap = OptimizationMap()
    optmap.add(gen_res, gen_res.get_parameters())
    optmap.add(gen_qubit, gen_qubit.get_parameters())
    opt = ScipyOptimizerGradient(measure_func=goal.get_value, measure_and_gradient_func=goal.get_value_and_gradient, optimization_map=optmap)
    opt.set_options({"maxfun": max_iter})
    optmap.register_params_with_optimizables()

.. code:: ipython3

    %%time
    opt.optimize(gen_res.tlist)


.. parsed-literal::

    Iteration   10 | Infid = 5.745168e-02


.. parsed-literal::

    Iteration   20 | Infid = 3.610521e-02


.. parsed-literal::

    Iteration   30 | Infid = 2.214250e-02


.. parsed-literal::

    Iteration   40 | Infid = 1.433837e-02


.. parsed-literal::

    Iteration   50 | Infid = 1.035732e-02


.. parsed-literal::

    Iteration   60 | Infid = 8.685263e-03


.. parsed-literal::

    Iteration   70 | Infid = 8.130577e-03


.. parsed-literal::

    Iteration   80 | Infid = 7.459853e-03


.. parsed-literal::

    Iteration   90 | Infid = 7.074717e-03


.. parsed-literal::

    Iteration  100 | Infid = 6.828301e-03


.. parsed-literal::

    CPU times: user 12min 32s, sys: 1.96 s, total: 12min 33s
    Wall time: 21.7 s




.. parsed-literal::

    {'status': 1, 'value': 0.0066150037700500874, 'iterations': 105, 'evaluations': 146, 'message': 'CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH'}



The new fidelities are

.. code:: ipython3

    print(f"Fidelity at N_T={n_fock_truncation_list[0]} = {meas_list[0].get_value(tlist)}")
    print(f"Fidelity at N_T={n_fock_truncation_list[1]} = {meas_list[1].get_value(tlist)}")


.. parsed-literal::

    Fidelity at N_T=15 = 0.9172701062233687
    Fidelity at N_T=20 = 0.9172703966730308


And the dynamics of the system under these optimized pulses looks like
the following

.. code:: ipython3

    plot_states_and_fock_number()



.. image:: 08B_Bosonic_grape_state_preparation_files/08B_Bosonic_grape_state_preparation_42_0.png


References
----------

- **(Khaneja et al., 2005)** N. Khaneja et al., “Optimal control of
  coupled spin dynamics: design of NMR pulse sequences by gradient
  ascent algorithms,” *Journal of Magnetic Resonance* **172**, 296–305
  (2005).
- **(Heeres et al., 2017)** R. W. Heeres et al., “Implementing a
  universal gate set on a logical qubit encoded in an oscillator,”
  *Nature Communications* **8**, 94 (2017).
- **(Eickbusch et al., 2022)** A. Eickbusch et al., “Fast universal
  control of an oscillator with weak dispersive coupling to a qubit,”
  *Nature Physics* **18**, 1464–1469 (2022).
- **(Blais et al., 2021)** A. Blais et al., “Circuit quantum
  electrodynamics,” *Reviews of Modern Physics* **93**, 025005 (2021).
