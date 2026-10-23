Arbitrary bosonic state preparation using smooth pulses: Gradient-based dCRAB optimization using GOAT over GRAPE method
=======================================================================================================================

In this example we solve the task in the previous example of preparing
an arbitrary bosonic state, but using smooth functions as the basis.
Here we perform a gradient-based dCRAB optimization, by using the
*GOAToverGRAPE* (combining GOAT (Machnes et al., 2018)
:cite:p:`machnes2018tunable` and GRAPE (Khaneja et al., 2005)
:cite:p:`khaneja2005optimal`) (or the GROUP (Sørensen et al., 2018)
:cite:p:`sorensen2018quantum`) method.

.. code:: ipython3

    import matplotlib.pyplot as plt
    import numpy as np
    
    from paraqeet.eom.schroedinger_equation import SchroedingerEquation
    from paraqeet.file_logger import FileLogger
    from paraqeet.hamiltonian.composite_hamiltonian import CompositeHamiltonian
    from paraqeet.hamiltonian.coupling import Coupling
    from paraqeet.hamiltonian.drive import Drive
    from paraqeet.hamiltonian.qubit import QubitHamiltonian
    from paraqeet.hamiltonian.resonator import ResonatorHamiltonian
    from paraqeet.measurement.fidelity import FidelityGRAPE
    from paraqeet.measurement.goat_over_grape import GOATOverGRAPE
    from paraqeet.measurement.weighted_sum_goal import WeightedSumGoal
    from paraqeet.optimization_map import OptimizationMap
    from paraqeet.optimizers.dcrab_optimizer_gradient import DCRABOptimizerGradient
    from paraqeet.propagation import GRAPE, Expm
    from paraqeet.quantity import Quantity
    from paraqeet.signal.envelopes import DCRABEnvelope
    from paraqeet.signal.pwc_generator import PWCGenerator
    from paraqeet.signal.signal import FlatTopGaussianFilter

Following the dCRAB optimization (Müller et al., 2022)
:cite:p:`muller2022one`, we initialize our pulses in the frequency
domain by using the ``DCRABEnvelope``. Further, we use a flat-top
Gaussian filter on top to ensure that the pulses start and end at zero.
And we use a ``PWCGenerator`` to generate the piece-wise constant signal
required for the ``GOAToverGRAPE`` method.

.. code:: ipython3

    # in seconds (See Eickbusch et al https://arxiv.org/abs/2111.06414 S9 A)
    delta_sampling = 33e-9
    n_pwc = 40  # number of piecewise constants in the pulse
    t_final = n_pwc * delta_sampling  # for now just set up for trying
    tlist = np.linspace(0, t_final, n_pwc + 1)
    eps_res = 3 * np.pi * 1.0  # initial amplitude of the resonator (in MHz)
    eps_max_res = 5 * eps_res  # maximum amplitude of the resonator (in MHz)
    eps_qubit = 3 * np.pi * 1.0  # initial amplitude of the qubit (in MHz)
    eps_max_qubit = 5 * eps_qubit  # maximum amplitude of the resonator (in MHz)
    
    tone_res = DCRABEnvelope(
        num_components=2,
        max_frequency=2 * np.pi * 2.0,
        amplitude=Quantity(eps_res * 1e6, -eps_max_res * 1e6, eps_max_res * 1e6, name="Amplitude CRAB resonator"),
        t_final=Quantity(t_final, 1 / 2 * t_final, 2 * t_final, name="t_final CRAB resonator"),
        seed=46874,
    )
    
    smooth_tone_res = FlatTopGaussianFilter(
        envelopes=tone_res, t_final=Quantity(t_final, 1 / 2 * t_final, 2 * t_final, name="t_final")
    )
    
    tone_qubit = DCRABEnvelope(
        num_components=2,
        max_frequency=2 * np.pi * 2.0,
        amplitude=Quantity(eps_qubit * 1e6, -eps_max_qubit * 1e6, eps_max_qubit * 1e6, name="Amplitude CRAB qubit"),
        t_final=Quantity(t_final, 1 / 2 * t_final, 2 * t_final, name="t_final CRAB qubit"),
        seed=189841,
    )
    
    smooth_tone_qubit = FlatTopGaussianFilter(
        envelopes=tone_qubit, t_final=Quantity(t_final, 1 / 2 * t_final, 2 * t_final, name="t_final")
    )
    
    gen_res = PWCGenerator(envelopes=[smooth_tone_res], tlist=tlist, max_amplitude=2 * eps_max_res * 1e6)
    gen_qubit = PWCGenerator(envelopes=[smooth_tone_qubit], tlist=tlist, max_amplitude=2 * eps_max_qubit * 1e6)

Here we plot the resonator tone and compare it to its smooth version.
Similarly one can plot the qubit tone.

.. code:: ipython3

    from plotting import plot_signal
    
    ts = np.linspace(0, t_final, 1001)
    fig, ax = plt.subplots(1, figsize=(5, 3))
    
    plot_signal(smooth_tone_res, ts, ax, linestyle="-", label="Smooth")
    plot_signal(gen_res, ts, ax, linestyle="--", label="PWC")
    
    plt.legend()
    plt.show()



.. image:: 08C_Bosonic_grape_with_smooth_pulses_files/08C_Bosonic_grape_with_smooth_pulses_5_0.png


Like the previous example, we define qubit and resonator parameters, as
well as the list of different Fock truncation numbers. We also set the
target Fock state.

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

Similar to the previous example, following (Heeres et al., 2017)
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
        dcrab_fid_list = []
    
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
                frechet_derivative=True,
            )
    
            prop_list.append(prop)
    
            fid = FidelityGRAPE(
                propagation_func=prop.get_value,
                propagation_and_gradient_func=prop.get_value_and_gradient,
                target_states=target_state,
                overlap=overlap_state_vector,
                fid=state_fidelity,
            )
            goat_over_grape_fid = GOATOverGRAPE(
                fid, generators=[gen_res, gen_qubit], propagation_resolution=propagation.resolution
            )
            dcrab_fid_list.append(goat_over_grape_fid)
    
        return initial_state_list, target_state_list, fock_number_op_list, prop_list, dcrab_fid_list
    
    
    initial_state_list, target_state_list, fock_number_op_list, prop_list, dcrab_meas_list = create_experiment(
        n_fock_truncation_list, fock_target
    )

We consider a cost function of the form

.. math::


   C(\varepsilon(t) ) = w_1 \sum_{N = N_{\mathrm{T}}^{(\mathrm{min})}}^{ N_{\mathrm{T}}^{(\mathrm{max})}} \mathcal{F}_{N} (\varepsilon(t) )   - \frac{w_2}{2} \sum_{N, N' = N_{\mathrm{T}}^{(\mathrm{min})}}^{ N_{\mathrm{T}}^{(\mathrm{max})}} \left[\mathcal{F}_{N}(\varepsilon(t))- \mathcal{F}_{N'} (\varepsilon(t) ) \right]^2,

that we want to maximize. This weighted cost function can be constructed
using the class WeightedSumGoal.

*Note - As we consider smooth pulses as our basis function, we do not
need to add additional smoothness cost to the goal function.*

.. code:: ipython3

    weights = np.array([1.0 for _ in range(len(dcrab_meas_list))])
    weight_sum_of_squares = -0.05
    total_sum_of_weights = np.sum(weights) + weight_sum_of_squares
    weights = weights / total_sum_of_weights
    weight_sum_of_squares = weight_sum_of_squares / total_sum_of_weights
    dcrab_meas_list_bool = [True for meas in dcrab_meas_list]
    sum_of_squares_options = {"weight": weight_sum_of_squares, "meas_bool": dcrab_meas_list_bool}
    
    dcrab_goal = WeightedSumGoal(dcrab_meas_list, weights, sum_of_squares_options=sum_of_squares_options)

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



.. image:: 08C_Bosonic_grape_with_smooth_pulses_files/08C_Bosonic_grape_with_smooth_pulses_13_0.png


And we compute the fidelities for the different truncation numbers

.. code:: ipython3

    print(f"Fidelity at N_T={n_fock_truncation_list[0]} = {dcrab_meas_list[0].get_value(tlist)}")
    print(f"Fidelity at N_T={n_fock_truncation_list[1]} = {dcrab_meas_list[1].get_value(tlist)}")


.. parsed-literal::

    Fidelity at N_T=3 = 0.0006103008293939482


.. parsed-literal::

    Fidelity at N_T=4 = 0.10317446818272527


which are quite poor! We now proceed with the pulse optimization.

Let’s define the parameters we want to optimize. Since, in this case, we
want to optimize the smooth pulses, the parameters for the optmap would
be the parameters of the ``DCRABEnvelope`` tone.

.. code:: ipython3

    params_tone_qubit = tone_qubit.get_parameters()
    params_tone_res = tone_res.get_parameters()
    params_tone_qubit




.. parsed-literal::

    [Amplitude CRAB qubit: 9.42e+06,
     t_final CRAB qubit: 1.32e-06,
     CRAB Re coefficient 0: 0.742,
     CRAB Re coefficient 1: -0.875,
     CRAB Re frequency 0: 1.8 Hz x 2pi,
     CRAB Re frequency 1: 522 mHz x 2pi,
     CRAB Re Phase 0: 2.15 rad,
     CRAB Re Phase 1: 1.93 rad,
     CRAB Im coefficient 0: 0.632,
     CRAB Im coefficient 1: -0.163,
     CRAB Im frequency 0: 1.68 Hz x 2pi,
     CRAB Im frequency 1: 331 mHz x 2pi,
     CRAB Im Phase 0: 1.25 rad,
     CRAB Im Phase 1: -645 mrad]



Further, we can write the optimization progress into a log file, which
we can later use to plot the infidelity vs function evaluation.

.. code:: ipython3

    import tempfile
    
    temp_dir = tempfile.TemporaryDirectory(suffix="pq")  # Ends in 'pq' to prevent _ at the end
    print(f"Logging directory = {temp_dir.name}")
    
    max_iter = 200  # set to 1000 for a good result; set to 10 for a quick example
    optmap = OptimizationMap()
    optmap.add(tone_res)
    optmap.add(tone_qubit)
    
    # Remove the t_final and  DRAG Delta from optimization from drag tone resonator and qubit
    optmap.remove(tone_qubit, params_tone_qubit[1])
    optmap.remove(tone_res, params_tone_res[1])
    optmap.register_params_with_optimizables()
    
    file_logger = FileLogger(temp_dir.name)
    
    # Define the optimizer
    opt = DCRABOptimizerGradient(
        measure_func=goat.get_value,
        measure_and_gradient_func=goat.get_value_and_gradient,
        optimization_map=optmap,
        super_iteration_every=100,
        max_super_iteration_num=2,
        print_every_iteration_num=50,
        super_iteration_tol=1e-6,
        seed=5648,
    )
    opt.set_options({"maxfun": max_iter, "workers": 32, "ftol": 1e-6})
    opt.logger = file_logger


.. parsed-literal::

    Logging directory = /tmp/tmp8rllmbropq


.. code:: ipython3

    optmap




.. parsed-literal::

    ==== <class 'paraqeet.signal.envelopes.DCRABEnvelope'> ====
    [Amplitude CRAB resonator: 9.42e+06, CRAB Re coefficient 0: 0.807, CRAB Re coefficient 1: -0.529, CRAB Re frequency 0: 1.51 Hz x 2pi, CRAB Re frequency 1: 916 mHz x 2pi, CRAB Re Phase 0: 780 mrad, CRAB Re Phase 1: -2.13 rad, CRAB Im coefficient 0: -0.614, CRAB Im coefficient 1: 0.803, CRAB Im frequency 0: 1.92 Hz x 2pi, CRAB Im frequency 1: 1.47 Hz x 2pi, CRAB Im Phase 0: -276 mrad, CRAB Im Phase 1: 1.86 rad]
    
    ==== <class 'paraqeet.signal.envelopes.DCRABEnvelope'> ====
    [Amplitude CRAB qubit: 9.42e+06, CRAB Re coefficient 0: 0.742, CRAB Re coefficient 1: -0.875, CRAB Re frequency 0: 1.8 Hz x 2pi, CRAB Re frequency 1: 522 mHz x 2pi, CRAB Re Phase 0: 2.15 rad, CRAB Re Phase 1: 1.93 rad, CRAB Im coefficient 0: 0.632, CRAB Im coefficient 1: -0.163, CRAB Im frequency 0: 1.68 Hz x 2pi, CRAB Im frequency 1: 331 mHz x 2pi, CRAB Im Phase 0: 1.25 rad, CRAB Im Phase 1: -645 mrad]




.. code:: ipython3

    %%time
    opt.optimize(gen_res.tlist)


.. parsed-literal::

    Iteration number = 0 	  Infidelity  = 8.910e-01


.. parsed-literal::

    Iteration number = 50 	  Infidelity  = 1.461e-01


.. parsed-literal::

    
    
    ==== Max iteration before a super-iteration reached ====
    ==== Starting super-iteration 1 ====
    *** Current lowest infidelity =  0.097 ***
    * Current no. of parameters = 50


.. parsed-literal::

    Iteration number = 100 	  Infidelity  = 5.445e-01


.. parsed-literal::

    Iteration number = 150 	  Infidelity  = 1.204e-01


.. parsed-literal::

    
    
    ==== Decrease in infidelity less than 1e-06 ====
    ==== Starting super-iteration 2 ====
    * Current lowest infidelity =  7.774e-02
    * Current no. of parameters = 74


.. parsed-literal::

    Stopping the optimization or backtracking to previous best fidelity. Going to step with 26 parameters.


.. parsed-literal::

    Setting parameters to the best values.
    CPU times: user 3min 22s, sys: 3.27 s, total: 3min 25s
    Wall time: 33.4 s


.. parsed-literal::

    Stopping the optimization or backtracking to previous best fidelity. Going to step with 50 parameters.




.. parsed-literal::

    {'status': 2, 'value': 0.07773963892059199, 'iterations': 197, 'evaluations': 261, 'message': '`callback` raised `StopIteration`.'}



Let’s set the optimization to the best parameters obtained during the
run

.. code:: ipython3

    opt.set_parameters(opt.best_params)




.. parsed-literal::

    [Amplitude CRAB resonator: -4.64e+07,
     CRAB Re coefficient 0: -0.406,
     CRAB Re coefficient 1: -0.892,
     CRAB Re coefficient 2: -0.627,
     CRAB Re coefficient 3: -0.999,
     CRAB Re coefficient 4: 0,
     CRAB Re coefficient 5: 0,
     CRAB Re frequency 0: 503 mHz x 2pi,
     CRAB Re frequency 1: 28.7 mHz x 2pi,
     CRAB Re frequency 2: 1.01 Hz x 2pi,
     CRAB Re frequency 3: 2 Hz x 2pi,
     CRAB Re frequency 4: 1 Hz x 2pi,
     CRAB Re frequency 5: 1 Hz x 2pi,
     CRAB Re Phase 0: -2.85 rad,
     CRAB Re Phase 1: -3.11 rad,
     CRAB Re Phase 2: -3.02 rad,
     CRAB Re Phase 3: 2.03 rad,
     CRAB Re Phase 4: 0 rad,
     CRAB Re Phase 5: 0 rad,
     CRAB Im coefficient 0: -0.949,
     CRAB Im coefficient 1: 0.535,
     CRAB Im coefficient 2: -0.89,
     CRAB Im coefficient 3: 0.521,
     CRAB Im coefficient 4: 0,
     CRAB Im coefficient 5: 0,
     CRAB Im frequency 0: 111 mHz x 2pi,
     CRAB Im frequency 1: 2 Hz x 2pi,
     CRAB Im frequency 2: 1.23 Hz x 2pi,
     CRAB Im frequency 3: 1.29 Hz x 2pi,
     CRAB Im frequency 4: 1 Hz x 2pi,
     CRAB Im frequency 5: 1 Hz x 2pi,
     CRAB Im Phase 0: -3.14 rad,
     CRAB Im Phase 1: 2.74 rad,
     CRAB Im phase 2: -1.15 rad,
     CRAB Im phase 3: -143 mrad,
     CRAB Im phase 4: 0 rad,
     CRAB Im phase 5: 0 rad,
     Amplitude CRAB qubit: 4.57e+07,
     CRAB Re coefficient 0: -0.0137,
     CRAB Re coefficient 1: -0.999,
     CRAB Re coefficient 2: 0.44,
     CRAB Re coefficient 3: 0.284,
     CRAB Re coefficient 4: 0,
     CRAB Re coefficient 5: 0,
     CRAB Re frequency 0: 2 Hz x 2pi,
     CRAB Re frequency 1: 675 mHz x 2pi,
     CRAB Re frequency 2: 729 mHz x 2pi,
     CRAB Re frequency 3: 1.96 Hz x 2pi,
     CRAB Re frequency 4: 1 Hz x 2pi,
     CRAB Re frequency 5: 1 Hz x 2pi,
     CRAB Re Phase 0: 3.13 rad,
     CRAB Re Phase 1: -491 mrad,
     CRAB Re Phase 2: 2.56 rad,
     CRAB Re Phase 3: 417 mrad,
     CRAB Re Phase 4: 0 rad,
     CRAB Re Phase 5: 0 rad,
     CRAB Im coefficient 0: 1,
     CRAB Im coefficient 1: 0.123,
     CRAB Im coefficient 2: 0.35,
     CRAB Im coefficient 3: 0.432,
     CRAB Im coefficient 4: 0,
     CRAB Im coefficient 5: 0,
     CRAB Im frequency 0: 881 mHz x 2pi,
     CRAB Im frequency 1: 431 mHz x 2pi,
     CRAB Im frequency 2: 159 mHz x 2pi,
     CRAB Im frequency 3: 1.12 Hz x 2pi,
     CRAB Im frequency 4: 1 Hz x 2pi,
     CRAB Im frequency 5: 1 Hz x 2pi,
     CRAB Im Phase 0: 489 mrad,
     CRAB Im Phase 1: -125 mrad,
     CRAB Im phase 2: -2.23 rad,
     CRAB Im phase 3: -1.39 rad,
     CRAB Im phase 4: 0 rad,
     CRAB Im phase 5: 0 rad]



We can now plot the new pulses and the corresponding dynamics. If the
number of iterations (max_iter) is chosen to be large enough, we should
see an improvement.

.. code:: ipython3

    plot_states_and_fock_number()



.. image:: 08C_Bosonic_grape_with_smooth_pulses_files/08C_Bosonic_grape_with_smooth_pulses_26_0.png


We can also compare the dynamics with the higher truncation number.
Ideally, if the pulse amplitude is not high enough to produce artifacts
due to the truncation of Hilbert space, the dynamics in the two cases
should match each other. Else one needs to either reduce the pulse
amplitude or increase the truncation cutoff.

.. code:: ipython3

    plot_states_and_fock_number(1)



.. image:: 08C_Bosonic_grape_with_smooth_pulses_files/08C_Bosonic_grape_with_smooth_pulses_28_0.png


The new fidelities are

.. code:: ipython3

    print(f"Fidelity at N_T={n_fock_truncation_list[0]} = {dcrab_meas_list[0].get_value(tlist)}")
    print(f"Fidelity at N_T={n_fock_truncation_list[1]} = {dcrab_meas_list[1].get_value(tlist)}")


.. parsed-literal::

    Fidelity at N_T=3 = 0.9190649779591606
    Fidelity at N_T=4 = 0.8794213071773376


For this small truncation number the dynamics and fidelities do not
match very well. This indicates the need for higher truncation numbers,
which we demonstrate in the next section.

Finally, let’s plot the variation of infidelity with evaluation number.

.. code:: ipython3

    from plotting import plot_infidelity_vs_evaluation_from_logs
    
    plot_infidelity_vs_evaluation_from_logs(log_path=temp_dir.name + "/opt.log", label="dCRAB optimization")




.. parsed-literal::

    <Axes: xlabel='Evaluation number', ylabel='Infidelity'>




.. image:: 08C_Bosonic_grape_with_smooth_pulses_files/08C_Bosonic_grape_with_smooth_pulses_33_1.png


Setting truncation to higher value
----------------------------------

Here we set the truncation values to 15 and 20 levels and rerun the
entire simulation. For a more realistic simulation, we advise the reader
to increase the truncation to 30 and 31 levels.

*Note - The following takes about 15 minutes to run on an AMD-EPYC Milan
processor with 64 cores.*

We first reset the generator parameters (by redefining them), and
redefine the resonator with higher truncation numbers.

.. code:: ipython3

    # in seconds (See Eickbusch et al https://arxiv.org/abs/2111.06414 S9 A)
    delta_sampling = 33e-9
    n_pwc = 40  # number of piecewise constants in the pulse
    t_final = n_pwc * delta_sampling  # for now just set up for trying
    tlist = np.linspace(0, t_final, n_pwc + 1)
    eps_res = 3 * np.pi * 1.0  # initial amplitude of the resonator (in MHz)
    eps_max_res = 5 * eps_res  # maximum amplitude of the resonator (in MHz)
    eps_qubit = 3 * np.pi * 1.0  # initial amplitude of the qubit (in MHz)
    eps_max_qubit = 5 * eps_qubit  # maximum amplitude of the resonator (in MHz)
    
    tone_res = DCRABEnvelope(
        num_components=3,
        max_frequency=2 * np.pi * 5.0,
        amplitude=Quantity(eps_res * 1e6, -eps_max_res * 1e6, eps_max_res * 1e6, name="Amplitude CRAB resonator"),
        t_final=Quantity(t_final, 1 / 2 * t_final, 2 * t_final, name="t_final CRAB resonator"),
        seed=17619276394480986,
    )
    
    smooth_tone_res = FlatTopGaussianFilter(
        envelopes=tone_res, t_final=Quantity(t_final, 1 / 2 * t_final, 2 * t_final, name="t_final")
    )
    
    tone_qubit = DCRABEnvelope(
        num_components=3,
        max_frequency=2 * np.pi * 5.0,
        amplitude=Quantity(eps_qubit * 1e6, -eps_max_qubit * 1e6, eps_max_qubit * 1e6, name="Amplitude CRAB qubit"),
        t_final=Quantity(t_final, 1 / 2 * t_final, 2 * t_final, name="t_final CRAB qubit"),
        seed=17619276394849436,
    )
    
    smooth_tone_qubit = FlatTopGaussianFilter(
        envelopes=tone_qubit, t_final=Quantity(t_final, 1 / 2 * t_final, 2 * t_final, name="t_final")
    )
    
    gen_res = PWCGenerator(envelopes=[smooth_tone_res], tlist=tlist, max_amplitude=2 * eps_max_res * 1e6)
    gen_qubit = PWCGenerator(envelopes=[smooth_tone_qubit], tlist=tlist, max_amplitude=2 * eps_max_qubit * 71e6)

.. code:: ipython3

    # Increasing the number of Fock states gives a more realistic scenario, at the price of increasing the simulation time
    n_fock_truncation_list = [15, 20]
    fock_target = 2
    n_times = 1001
    
    omega_range_factor = 1e-3  # just for initialization of the Quantity object
    omega_res = 2 * np.pi * 5.26e9
    omega_qubit = 2 * np.pi * 6.65e9
    
    omega_drive_res = omega_res
    omega_drive_qubit = omega_qubit
    
    detuning_drive_res = omega_res - omega_drive_res
    detuning_drive_qubit = omega_qubit - omega_drive_qubit
    
    # Create a new experiment and redefine the measurement
    initial_state_list, target_state_list, fock_number_op_list, prop_list, dcrab_meas_list = create_experiment(
        n_fock_truncation_list, fock_target
    )
    
    weights = np.array([1.0 for _ in range(len(dcrab_meas_list))])
    weight_sum_of_squares = -0.05
    total_sum_of_weights = np.sum(weights) + weight_sum_of_squares
    weights = weights / total_sum_of_weights
    weight_sum_of_squares = weight_sum_of_squares / total_sum_of_weights
    dcrab_meas_list_bool = [True for meas in dcrab_meas_list]
    sum_of_squares_options = {"weight": weight_sum_of_squares, "meas_bool": dcrab_meas_list_bool}
    
    dcrab_goal = WeightedSumGoal(dcrab_meas_list, weights, sum_of_squares_options=sum_of_squares_options)

.. code:: ipython3

    plot_states_and_fock_number()



.. image:: 08C_Bosonic_grape_with_smooth_pulses_files/08C_Bosonic_grape_with_smooth_pulses_37_0.png


.. code:: ipython3

    plot_states_and_fock_number(1)



.. image:: 08C_Bosonic_grape_with_smooth_pulses_files/08C_Bosonic_grape_with_smooth_pulses_38_0.png


Initial fidelity before optimization

.. code:: ipython3

    print(f"Fidelity at N_T={n_fock_truncation_list[0]} = {dcrab_meas_list[0].get_value(tlist)}")
    print(f"Fidelity at N_T={n_fock_truncation_list[1]} = {dcrab_meas_list[1].get_value(tlist)}")


.. parsed-literal::

    Fidelity at N_T=15 = 0.012497050007112244


.. parsed-literal::

    Fidelity at N_T=20 = 0.012497050007112228


.. code:: ipython3

    tone_res.get_parameters()




.. parsed-literal::

    [Amplitude CRAB resonator: 9.42e+06,
     t_final CRAB resonator: 1.32e-06,
     CRAB Re coefficient 0: -0.218,
     CRAB Re coefficient 1: -0.711,
     CRAB Re coefficient 2: -0.223,
     CRAB Re frequency 0: 4.77 Hz x 2pi,
     CRAB Re frequency 1: 4.34 Hz x 2pi,
     CRAB Re frequency 2: 4.59 Hz x 2pi,
     CRAB Re Phase 0: 3 rad,
     CRAB Re Phase 1: 32.9 mrad,
     CRAB Re Phase 2: 2.61 rad,
     CRAB Im coefficient 0: -0.725,
     CRAB Im coefficient 1: -0.262,
     CRAB Im coefficient 2: 0.901,
     CRAB Im frequency 0: 3.33 Hz x 2pi,
     CRAB Im frequency 1: 3.18 Hz x 2pi,
     CRAB Im frequency 2: 1.97 Hz x 2pi,
     CRAB Im Phase 0: 1.29 rad,
     CRAB Im Phase 1: 1.45 rad,
     CRAB Im Phase 2: 378 mrad]



.. code:: ipython3

    tone_qubit.get_parameters()




.. parsed-literal::

    [Amplitude CRAB qubit: 9.42e+06,
     t_final CRAB qubit: 1.32e-06,
     CRAB Re coefficient 0: -0.693,
     CRAB Re coefficient 1: 0.833,
     CRAB Re coefficient 2: 0.687,
     CRAB Re frequency 0: 4.03 Hz x 2pi,
     CRAB Re frequency 1: 2.46 Hz x 2pi,
     CRAB Re frequency 2: 4.43 Hz x 2pi,
     CRAB Re Phase 0: -617 mrad,
     CRAB Re Phase 1: -2.14 rad,
     CRAB Re Phase 2: -1.32 rad,
     CRAB Im coefficient 0: -0.152,
     CRAB Im coefficient 1: -0.103,
     CRAB Im coefficient 2: 0.579,
     CRAB Im frequency 0: 2.09 Hz x 2pi,
     CRAB Im frequency 1: 4.5 Hz x 2pi,
     CRAB Im frequency 2: 2.08 Hz x 2pi,
     CRAB Im Phase 0: -2.22 rad,
     CRAB Im Phase 1: 3.02 rad,
     CRAB Im Phase 2: -103 mrad]



Redefine the optmap and the optimizer and rerun the optimization

.. code:: ipython3

    import tempfile
    
    temp_dir = tempfile.TemporaryDirectory(suffix="pq")  # Ends in 'pq' to prevent _ at the end
    print(f"Logging directory = {temp_dir.name}")
    
    max_iter = 300  # set to 1000 for a good result; set to 10 for a quick example
    optmap = OptimizationMap()
    optmap.add(tone_res)
    optmap.add(tone_qubit)
    
    # Remove the t_final and  DRAG Delta from optimization from drag tone resonator and qubit
    optmap.remove(tone_qubit, params_tone_qubit[1])
    optmap.remove(tone_res, params_tone_res[1])
    optmap.register_params_with_optimizables()
    
    file_logger = FileLogger(temp_dir.name)
    
    # Define the optimizer
    opt = DCRABOptimizerGradient(
        measure_func=goat.get_value,
        measure_and_gradient_func=goat.get_value_and_gradient,
        optimization_map=optmap,
        super_iteration_every=100,
        max_super_iteration_num=3,  # Increase the number of super iterations for a better result
        print_every_iteration_num=10,
        super_iteration_tol=1e-6,
        seed=8647,
    )
    opt.set_options({"maxfun": max_iter, "workers": 32})
    opt.logger = file_logger


.. parsed-literal::

    Logging directory = /tmp/tmpzgq1fdx_pq


.. parsed-literal::

    Implicitly cleaning up <TemporaryDirectory '/tmp/tmp8rllmbropq'>


.. code:: ipython3

    %%time
    opt.optimize(gen_res.tlist)
    opt.set_parameters(opt.best_params)


.. parsed-literal::

    Iteration number = 0 	  Infidelity  = 8.764e-01


.. parsed-literal::

    Iteration number = 10 	  Infidelity  = 6.840e-01


.. parsed-literal::

    Iteration number = 20 	  Infidelity  = 3.873e-01


.. parsed-literal::

    Iteration number = 30 	  Infidelity  = 3.186e-01


.. parsed-literal::

    Iteration number = 40 	  Infidelity  = 2.724e-01


.. parsed-literal::

    Iteration number = 50 	  Infidelity  = 2.298e-01


.. parsed-literal::

    Iteration number = 60 	  Infidelity  = 2.004e-01


.. parsed-literal::

    Iteration number = 70 	  Infidelity  = 1.874e-01


.. parsed-literal::

    Iteration number = 80 	  Infidelity  = 1.662e-01


.. parsed-literal::

    Iteration number = 90 	  Infidelity  = 1.584e-01


.. parsed-literal::

    
    
    ==== Max iteration before a super-iteration reached ====
    ==== Starting super-iteration 1 ====
    *** Current lowest infidelity =  0.154 ***
    * Current no. of parameters = 74


.. parsed-literal::

    Iteration number = 100 	  Infidelity  = 6.616e-01


.. parsed-literal::

    Iteration number = 110 	  Infidelity  = 1.805e-01


.. parsed-literal::

    Iteration number = 120 	  Infidelity  = 1.318e-01


.. parsed-literal::

    Iteration number = 130 	  Infidelity  = 1.046e-01


.. parsed-literal::

    Iteration number = 140 	  Infidelity  = 8.466e-02


.. parsed-literal::

    Iteration number = 150 	  Infidelity  = 7.465e-02


.. parsed-literal::

    Iteration number = 160 	  Infidelity  = 6.705e-02


.. parsed-literal::

    Iteration number = 170 	  Infidelity  = 6.256e-02


.. parsed-literal::

    Iteration number = 180 	  Infidelity  = 5.763e-02


.. parsed-literal::

    Iteration number = 190 	  Infidelity  = 5.350e-02


.. parsed-literal::

    
    
    ==== Max iteration before a super-iteration reached ====
    ==== Starting super-iteration 2 ====
    *** Current lowest infidelity =  0.052 ***
    * Current no. of parameters = 110


.. parsed-literal::

    Iteration number = 200 	  Infidelity  = 5.294e-01


.. parsed-literal::

    Iteration number = 210 	  Infidelity  = 2.062e-01


.. parsed-literal::

    Iteration number = 220 	  Infidelity  = 1.483e-01


.. parsed-literal::

    Iteration number = 230 	  Infidelity  = 8.268e-02


.. parsed-literal::

    Iteration number = 240 	  Infidelity  = 7.019e-02


.. parsed-literal::

    Iteration number = 250 	  Infidelity  = 5.853e-02


.. parsed-literal::

    Iteration number = 260 	  Infidelity  = 5.537e-02


.. parsed-literal::

    Iteration number = 270 	  Infidelity  = 5.373e-02


.. parsed-literal::

    Iteration number = 280 	  Infidelity  = 5.184e-02


.. parsed-literal::

    Iteration number = 290 	  Infidelity  = 5.012e-02


.. parsed-literal::

    
    
    ==== Max iteration before a super-iteration reached ====
    ==== Starting super-iteration 3 ====
    *** Current lowest infidelity =  0.048 ***
    * Current no. of parameters = 146


.. parsed-literal::

    Iteration number = 300 	  Infidelity  = 5.472e-01


.. parsed-literal::

    Stopping the optimization or backtracking to previous best fidelity. Going to step with 110 parameters.


.. parsed-literal::

    Stopping the optimization or backtracking to previous best fidelity. Going to step with 74 parameters.


.. parsed-literal::

    Stopping the optimization or backtracking to previous best fidelity. Going to step with 38 parameters.


.. parsed-literal::

    Setting parameters to the best values.
    CPU times: user 3h 54min 54s, sys: 1min 25s, total: 3h 56min 20s
    Wall time: 6min 12s




.. parsed-literal::

    [Amplitude CRAB resonator: 4.69e+07,
     CRAB Re coefficient 0: 0.00318,
     CRAB Re coefficient 1: -0.789,
     CRAB Re coefficient 2: 0.00219,
     CRAB Re coefficient 3: -0.13,
     CRAB Re coefficient 4: 0.0278,
     CRAB Re coefficient 5: -0.907,
     CRAB Re coefficient 6: 0.0271,
     CRAB Re coefficient 7: 0.0776,
     CRAB Re coefficient 8: -0.0146,
     CRAB Re coefficient 9: 0,
     CRAB Re coefficient 10: 0,
     CRAB Re coefficient 11: 0,
     CRAB Re frequency 0: 4.73 Hz x 2pi,
     CRAB Re frequency 1: 4.77 Hz x 2pi,
     CRAB Re frequency 2: 4.67 Hz x 2pi,
     CRAB Re frequency 3: 3.21 Hz x 2pi,
     CRAB Re frequency 4: 1.75 Hz x 2pi,
     CRAB Re frequency 5: 3.06 Hz x 2pi,
     CRAB Re frequency 6: 1.61 Hz x 2pi,
     CRAB Re frequency 7: 1.18 Hz x 2pi,
     CRAB Re frequency 8: 477 mHz x 2pi,
     CRAB Re frequency 9: 2.5 Hz x 2pi,
     CRAB Re frequency 10: 2.5 Hz x 2pi,
     CRAB Re frequency 11: 2.5 Hz x 2pi,
     CRAB Re Phase 0: 3.02 rad,
     CRAB Re Phase 1: -753 mrad,
     CRAB Re Phase 2: 2.75 rad,
     CRAB Re Phase 3: -586 mrad,
     CRAB Re Phase 4: -1.04 rad,
     CRAB Re Phase 5: 1.63 rad,
     CRAB Re Phase 6: -1.79 rad,
     CRAB Re Phase 7: 2.77 rad,
     CRAB Re Phase 8: -1.78 rad,
     CRAB Re Phase 9: 0 rad,
     CRAB Re Phase 10: 0 rad,
     CRAB Re Phase 11: 0 rad,
     CRAB Im coefficient 0: -0.986,
     CRAB Im coefficient 1: -0.41,
     CRAB Im coefficient 2: 0.765,
     CRAB Im coefficient 3: 0.684,
     CRAB Im coefficient 4: 0.074,
     CRAB Im coefficient 5: 0.109,
     CRAB Im coefficient 6: 0.211,
     CRAB Im coefficient 7: 0.0404,
     CRAB Im coefficient 8: -0.31,
     CRAB Im coefficient 9: 0,
     CRAB Im coefficient 10: 0,
     CRAB Im coefficient 11: 0,
     CRAB Im frequency 0: 3.42 Hz x 2pi,
     CRAB Im frequency 1: 3.46 Hz x 2pi,
     CRAB Im frequency 2: 1.44 Hz x 2pi,
     CRAB Im frequency 3: 2.94 Hz x 2pi,
     CRAB Im frequency 4: 4.96 Hz x 2pi,
     CRAB Im frequency 5: 3.63 Hz x 2pi,
     CRAB Im frequency 6: 3.58 Hz x 2pi,
     CRAB Im frequency 7: 3.78 Hz x 2pi,
     CRAB Im frequency 8: 3.17 Hz x 2pi,
     CRAB Im frequency 9: 2.5 Hz x 2pi,
     CRAB Im frequency 10: 2.5 Hz x 2pi,
     CRAB Im frequency 11: 2.5 Hz x 2pi,
     CRAB Im Phase 0: 3.14 rad,
     CRAB Im Phase 1: 3 rad,
     CRAB Im Phase 2: 545 mrad,
     CRAB Im phase 3: 1.88 rad,
     CRAB Im phase 4: 683 mrad,
     CRAB Im phase 5: -980 mrad,
     CRAB Im phase 6: -1.2 rad,
     CRAB Im phase 7: 1.72 rad,
     CRAB Im phase 8: -1.79 rad,
     CRAB Im phase 9: 0 rad,
     CRAB Im phase 10: 0 rad,
     CRAB Im phase 11: 0 rad,
     Amplitude CRAB qubit: 2.91e+07,
     CRAB Re coefficient 0: -0.981,
     CRAB Re coefficient 1: 0.832,
     CRAB Re coefficient 2: 0.432,
     CRAB Re coefficient 3: 0.279,
     CRAB Re coefficient 4: -0.27,
     CRAB Re coefficient 5: 0.0849,
     CRAB Re coefficient 6: -0.148,
     CRAB Re coefficient 7: -0.21,
     CRAB Re coefficient 8: -0.0741,
     CRAB Re coefficient 9: 0,
     CRAB Re coefficient 10: 0,
     CRAB Re coefficient 11: 0,
     CRAB Re frequency 0: 3.98 Hz x 2pi,
     CRAB Re frequency 1: 4.06 Hz x 2pi,
     CRAB Re frequency 2: 4.49 Hz x 2pi,
     CRAB Re frequency 3: 4.99 Hz x 2pi,
     CRAB Re frequency 4: 1.24 Hz x 2pi,
     CRAB Re frequency 5: 4.18 Hz x 2pi,
     CRAB Re frequency 6: 3.48 Hz x 2pi,
     CRAB Re frequency 7: 1.52 Hz x 2pi,
     CRAB Re frequency 8: 3.39 Hz x 2pi,
     CRAB Re frequency 9: 2.5 Hz x 2pi,
     CRAB Re frequency 10: 2.5 Hz x 2pi,
     CRAB Re frequency 11: 2.5 Hz x 2pi,
     CRAB Re Phase 0: -492 mrad,
     CRAB Re Phase 1: -2.37 rad,
     CRAB Re Phase 2: 318 mrad,
     CRAB Re Phase 3: 2.46 rad,
     CRAB Re Phase 4: -2.31 rad,
     CRAB Re Phase 5: -3.14 rad,
     CRAB Re Phase 6: -1.11 rad,
     CRAB Re Phase 7: -1.24 rad,
     CRAB Re Phase 8: 2.53 rad,
     CRAB Re Phase 9: 0 rad,
     CRAB Re Phase 10: 0 rad,
     CRAB Re Phase 11: 0 rad,
     CRAB Im coefficient 0: -0.154,
     CRAB Im coefficient 1: 0.0371,
     CRAB Im coefficient 2: 0.756,
     CRAB Im coefficient 3: 0.293,
     CRAB Im coefficient 4: -0.072,
     CRAB Im coefficient 5: -0.32,
     CRAB Im coefficient 6: 0.38,
     CRAB Im coefficient 7: -0.0912,
     CRAB Im coefficient 8: 0.391,
     CRAB Im coefficient 9: 0,
     CRAB Im coefficient 10: 0,
     CRAB Im coefficient 11: 0,
     CRAB Im frequency 0: 2.99 Hz x 2pi,
     CRAB Im frequency 1: 4.93 Hz x 2pi,
     CRAB Im frequency 2: 2.3 Hz x 2pi,
     CRAB Im frequency 3: 4.3 Hz x 2pi,
     CRAB Im frequency 4: 2.08 Hz x 2pi,
     CRAB Im frequency 5: 1.48 Hz x 2pi,
     CRAB Im frequency 6: 3.12 Hz x 2pi,
     CRAB Im frequency 7: 4.95 Hz x 2pi,
     CRAB Im frequency 8: 3.51 Hz x 2pi,
     CRAB Im frequency 9: 2.5 Hz x 2pi,
     CRAB Im frequency 10: 2.5 Hz x 2pi,
     CRAB Im frequency 11: 2.5 Hz x 2pi,
     CRAB Im Phase 0: -1.65 rad,
     CRAB Im Phase 1: 3.13 rad,
     CRAB Im Phase 2: 424 mrad,
     CRAB Im phase 3: 2.32 rad,
     CRAB Im phase 4: -1.34 rad,
     CRAB Im phase 5: 3.06 rad,
     CRAB Im phase 6: 1.35 rad,
     CRAB Im phase 7: 622 mrad,
     CRAB Im phase 8: -1.43 rad,
     CRAB Im phase 9: 0 rad,
     CRAB Im phase 10: 0 rad,
     CRAB Im phase 11: 0 rad]



And the new fidelities are

.. code:: ipython3

    print(f"Fidelity at N_T={n_fock_truncation_list[0]} = {dcrab_meas_list[0].get_value(tlist)}")
    print(f"Fidelity at N_T={n_fock_truncation_list[1]} = {dcrab_meas_list[1].get_value(tlist)}")


.. parsed-literal::

    Fidelity at N_T=15 = 0.9393575641534029
    Fidelity at N_T=20 = 0.9168231114039078


Here the dynamics and the fidelities are very similar to one another,
indicating no truncation artifacts.

Let’s plot the optimized pulses and dynamics

.. code:: ipython3

    plot_states_and_fock_number()



.. image:: 08C_Bosonic_grape_with_smooth_pulses_files/08C_Bosonic_grape_with_smooth_pulses_50_0.png


And verify that the dynamics with the higher truncation look the same

.. code:: ipython3

    plot_states_and_fock_number(1)



.. image:: 08C_Bosonic_grape_with_smooth_pulses_files/08C_Bosonic_grape_with_smooth_pulses_52_0.png


Finally, let’s plot the variation of infidelity with evaluation number.

.. code:: ipython3

    from plotting import plot_infidelity_vs_evaluation_from_logs
    
    plot_infidelity_vs_evaluation_from_logs(log_path=temp_dir.name + "/opt.log", label="dCRAB optimization")




.. parsed-literal::

    <Axes: xlabel='Evaluation number', ylabel='Infidelity'>




.. image:: 08C_Bosonic_grape_with_smooth_pulses_files/08C_Bosonic_grape_with_smooth_pulses_54_1.png


The sharp rise in infidelity represents the beginning of a
super-iteration

References
----------

- **(Sørensen et al., 2018)** J. J. W. H. Sørensen et al., “Quantum
  optimal control in a chopped basis: Applications in control of
  Bose-Einstein condensates,” *Physical Review A* **98**, 022119 (2018).
- **(Müller et al., 2022)** M. M. Müller et al., “One decade of quantum
  optimal control in the chopped random basis,” *Reports on Progress in
  Physics* **85**, 076001 (2022).
- **(Heeres et al., 2017)** R. W. Heeres et al., “Implementing a
  universal gate set on a logical qubit encoded in an oscillator,”
  *Nature Communications* **8**, 94 (2017).
- **(Khaneja et al., 2005)** N. Khaneja et al., “Optimal control of
  coupled spin dynamics: design of NMR pulse sequences by gradient
  ascent algorithms,” *Journal of Magnetic Resonance* **172**, 296–305
  (2005).
- **(Machnes et al., 2018)** S. Machnes et al., “Tunable, flexible, and
  efficient optimization of control pulses for practical qubits,”
  *Physical Review Letters* **120**, 150401 (2018).
