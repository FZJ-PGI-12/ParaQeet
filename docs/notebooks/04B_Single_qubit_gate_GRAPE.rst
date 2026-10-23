Single spin Part 3: Single qubit gate optimization using GRAPE
==============================================================

1. Generate a PWC pulse shape
-----------------------------

.. code:: ipython3

    import matplotlib.pyplot as plt
    import numpy as np
    
    from paraqeet.eom.schroedinger_equation import SchroedingerEquation
    from paraqeet.hamiltonian.drive import Drive
    from paraqeet.hamiltonian.transmon import TransmonHamiltonian
    from paraqeet.quantity import Quantity
    from paraqeet.signal.envelopes import GaussEnvelope
    from paraqeet.signal.pwc_generator import PWCGenerator

First, let’s generate a piecewise constant (PWC) pulse envelope for the
Gaussian pulse

.. code:: ipython3

    t_final = 20e-9
    tlist = np.linspace(0, t_final, 101)
    tone = GaussEnvelope(amplitude=Quantity(2 * np.pi / t_final / 3, -5 * np.pi / t_final, 5 * np.pi / t_final))
    tone.t_final.set_value(t_final)
    gen = PWCGenerator(envelopes=[tone], tlist=tlist)
    gen.multiply_flat_top = True

.. code:: ipython3

    params = gen.get_parameters()

.. code:: ipython3

    from plotting import plot_signal
    
    ts = np.linspace(0, t_final, 501)
    fig, ax = plt.subplots(1, figsize=(5, 3))
    plot_signal(tone, ts, ax, linestyle="-", label="Smooth")
    plot_signal(gen, ts, ax, linestyle="--", label="PWC")
    ax.legend(loc=1, frameon=True)
    plt.show()



.. image:: 04B_Single_qubit_gate_GRAPE_files/04B_Single_qubit_gate_GRAPE_6_0.png


2. Define Hamiltonian in the rotating frame of drive
----------------------------------------------------

Next, we set up the qubit system we want to control. We define the
Hamiltonian in the rotating frame of drive such that the pulse
oscillates slowly to apply GRAPE (Khaneja et al., 2005)
:cite:p:`khaneja2005optimal` gradients.

The Hamiltonian in the rotating frame of the drive is given by -

.. math::  H(t) = \big(\omega_q - \omega_d\big) b^\dagger b -\frac{\alpha}{2} (b^\dagger)^2 b^2 + (\epsilon(t) b + \epsilon(t)^* b)

.. code:: ipython3

    freq = 7.86e9 * 2 * np.pi
    num_levels = 3
    anharm = -50e6 * 2 * np.pi
    offset = 5e6 * 2 * np.pi
    
    drive_freq = freq + offset
    qubit_freq = freq - drive_freq
    
    transmon_hamiltonian = TransmonHamiltonian(
        frequency=Quantity(
            qubit_freq,
            1.2 * qubit_freq,
            0.8 * qubit_freq,
            unit="Hz",
            name="Frequency",
        ),
        anharmonicity=Quantity(anharm, 1.2 * anharm, 0.8 * anharm, unit="Hz", name="Anharmonicity"),
        drives=[],
        num_levels=num_levels,
    )
    
    drive = Drive(transmon_hamiltonian.annihilation_op, gen, add_hermitian=True)
    transmon_hamiltonian.drives = [drive]
    
    
    model = SchroedingerEquation(
        hamiltonian_func=transmon_hamiltonian.get_value, hamiltonian_gradient_func=transmon_hamiltonian.get_gradient
    )

.. code:: ipython3

    from paraqeet.measurement.fidelity import FidelityGRAPE
    from paraqeet.measurement.utils import gate_fidelity, overlap_state_vector
    from paraqeet.propagation import GRAPE, Expm
    from paraqeet.propagation.utils import grape_operator_sandwich_function_closed
    
    init = np.array([[1.0], [0.0], [0.0]])  # |0>
    target = np.array([[0.0], [1.0], [0.0]])  # |1>
    
    times = np.array([0.0, t_final])
    
    propagation = Expm(eom_func=model.get_value, resolution=1e9, initial_state=init)
    prop = GRAPE(
        propagation,
        eom_gradient_func=model.get_gradient,
        target_state=target,
        operator_sandwich_function=grape_operator_sandwich_function_closed,
        order=3,
    )
    
    zeroone = FidelityGRAPE(
        propagation_func=prop.get_value,
        propagation_and_gradient_func=prop.get_value_and_gradient,
        target_states=target,
        overlap=overlap_state_vector,
        fid=gate_fidelity,
    )

.. code:: ipython3

    from plotting import plot_signal_and_dynamics
    
    ts = np.linspace(0.0, t_final, 101)
    plot_signal_and_dynamics(gen, prop, ts, state_labels=[r"$|0\rangle$", r"$|1\rangle$"])




.. parsed-literal::

    array([<Axes: ylabel='Amplitude \n[MHz / $2\\pi$]'>, <Axes: xlabel='Time [ns]', ylabel='Population'>], dtype=object)




.. image:: 04B_Single_qubit_gate_GRAPE_files/04B_Single_qubit_gate_GRAPE_11_1.png


As expected, we get a partial transfer and a low fidelity.

.. code:: ipython3

    zeroone.get_value(times)




.. parsed-literal::

    Array(0.31, dtype=float64)



3. Optimization
---------------

We define an optimizer and link our fidelity measure as a goal function
and the parameters of the cosine tone and optimize just amplitude and
frequency, as in the state transfer example.

.. code:: ipython3

    from paraqeet.optimization_map import OptimizationMap
    from paraqeet.optimizers.scipy_optimizer import ScipyOptimizerGradient
    
    optmap = OptimizationMap()
    optmap.add(gen, params)
    opt = ScipyOptimizerGradient(measure_and_gradient_func=zeroone.get_value_and_gradient, optimization_map=optmap)

.. code:: ipython3

    opt.optimize(gen.tlist)


.. parsed-literal::

    Iteration   10 | Infid = 1.640602e-08




.. parsed-literal::

    {'status': 1, 'value': 1.5388084140255387e-09, 'iterations': 11, 'evaluations': 14, 'message': 'CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL'}



.. code:: ipython3

    plot_signal_and_dynamics(gen, prop, ts, state_labels=[r"$|0\rangle$", r"$|1\rangle$"])




.. parsed-literal::

    array([<Axes: ylabel='Amplitude \n[MHz / $2\\pi$]'>, <Axes: xlabel='Time [ns]', ylabel='Population'>], dtype=object)




.. image:: 04B_Single_qubit_gate_GRAPE_files/04B_Single_qubit_gate_GRAPE_17_1.png


References
----------

- **(Khaneja et al., 2005)** N. Khaneja et al., “Optimal control of
  coupled spin dynamics: design of NMR pulse sequences by gradient
  ascent algorithms,” *Journal of Magnetic Resonance* **172**, 296–305
  (2005).
