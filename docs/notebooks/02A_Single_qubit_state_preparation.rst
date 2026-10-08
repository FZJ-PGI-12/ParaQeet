State preparation of a single spin
==================================

In this notebook, we will study a simple problem of state preparation
for a single spin or qubit.

.. code:: ipython3

    import jax.numpy as jnp
    import numpy as np
    
    from paraqeet import Fidelity
    from paraqeet.eom.schroedinger_equation import SchroedingerEquation
    from paraqeet.hamiltonian.drive import Drive
    from paraqeet.hamiltonian.qubit import QubitHamiltonian
    from paraqeet.measurement.utils import gate_fidelity, overlap_state_vector
    from paraqeet.optimization_map import OptimizationMap
    from paraqeet.optimizers.scipy_optimizer_gradient import ScipyOptimizerGradient
    from paraqeet.quantity import Quantity
    from paraqeet.signal.envelopes import ConstantEnvelope
    from paraqeet.signal.iq_mixer import IQMixer

System Setup
------------

We first set up the qubit system we want to control. We set the qubit
frequency :math:`\omega_q / 2 \pi` to be :math:`4.8` GHz and define the
Hamiltonian as

.. math:: H(t)=H_\text{drift}+H_c(t)= \frac{\omega_q}{2} \sigma_z + \Omega(t)\sigma_x, 

\ where :math:`\Omega(t)` will be supplied by the generator.

.. code:: ipython3

    freq_q = 4.8e9
    omega_q = 2 * np.pi * freq_q
    
    qubit_hamiltonian = QubitHamiltonian(frequency=Quantity(omega_q, 0.8 * omega_q, 1.2 * omega_q, unit="Hz"), drives=[])

For signal generation, we define a simple cosine shaped tone generator
:math:`A \cos(\omega t)`

.. code:: ipython3

    envelope = ConstantEnvelope()
    gen = IQMixer(envelopes=[envelope])

We can inspect the pre-defined parameters with

.. code:: ipython3

    params_tone = envelope.get_parameters()
    print(params_tone)
    params_gen = gen.get_parameters()
    print(params_gen)


.. parsed-literal::

    [Amplitude: 24.7 MHz x 2pi, t_final: 32 ns]
    [Amplitude: 24.7 MHz x 2pi, t_final: 32 ns, lo_freq: 4.8 GHz x 2pi, Phase: 0 rad]


In this notebook, we would like to optimize the amplitude ``Amplitude``
and frequency ``lo_freq`` of the drive. We add a drive on the qubit.

.. code:: ipython3

    drive = Drive(qubit_hamiltonian.sigma_x, gen)
    qubit_hamiltonian.drives = [drive]
    schrgl = SchroedingerEquation(
        hamiltonian_func=qubit_hamiltonian.get_value,
        hamiltonian_gradient_func=qubit_hamiltonian.get_gradient,
    )

Textbook values for implementing an :math:`X` rotation on this system at
a time :math:`T` would be :math:`\omega=\omega_q` and :math:`A=\pi/T`.
We use some offset from these values as an initial guess to demonstrate
the optimization procedure.

.. code:: ipython3

    t_simu = 10e-9
    params_gen[0].set_value(0.8 * np.pi / t_simu)
    params_gen[2].set_value(1.01 * omega_q)

It is important to note that, although we changed the parameters of
``params_gen`` the corresponding parameter of ``params_tone`` also
changes, due to Python’s “Pass By Object Reference” scheme. In fact,

.. code:: ipython3

    print(params_tone)
    print(params_gen)


.. parsed-literal::

    [Amplitude: 40 MHz x 2pi, t_final: 32 ns]
    [Amplitude: 40 MHz x 2pi, t_final: 32 ns, lo_freq: 4.85 GHz x 2pi, Phase: 0 rad]


We can try to see what happens if we print the gradient of the
controlled qubit at the time ``t_simu``:

.. code:: ipython3

    print(qubit_hamiltonian.get_value_and_gradient(np.array([t_simu])))


.. parsed-literal::

    (Array([[[-1.508e+10, -2.493e+08],
            [-2.493e+08,  1.508e+10]]], dtype=float64), Array([], shape=(1, 0, 2, 2), dtype=float64))


We see that in this case it is empty. This is because we haven’t yet
defined an ``OptimizationMap`` object that defines the optimizable
parameters. Also note that the tone parameter ``t_final`` is simply the
time after which the pulse is assumed to be zero and it is set by
default to :math:`32 \, \mathrm{ns}`. This is not necessarily the
simulation time, which is another parameter of our choice called
``t_simu`` in this case.

Next, we select a propagation method, piecewise constant exponentiation,
and configure a state transfer problem from :math:`\ket{0}` to
:math:`\ket{1}`.

.. code:: ipython3

    from paraqeet.propagation.auto_diff_gradients import AutoDiffGradients
    from paraqeet.propagation.expm import Expm
    
    init = np.array([[1.0], [0.0]])  # |0>
    target = np.array([[0.0], [1.0]])  # |1>
    
    prop = Expm(eom_func=schrgl.get_value, resolution=100e9, initial_state=init)  # implicit timestep is 1 / resolution
    times = np.array([0.0, t_simu])
    
    prop_AD = AutoDiffGradients(propagation=prop, eom_gradient_func=schrgl.get_gradient)
    
    zeroone = Fidelity(
        propagation_func=prop_AD.get_value,
        propagation_and_gradient_func=prop_AD.get_value_and_gradient,
        target_states=target,
        overlap=overlap_state_vector,
        fid=gate_fidelity,
    )

Population dynamics
-------------------

.. code:: ipython3

    from plotting import plot_signal_and_dynamics
    
    ts = np.linspace(0.0, t_simu, 101)
    plot_signal_and_dynamics(gen, prop, ts, state_labels=[r"$|0\rangle$", r"$|1\rangle$"])




.. parsed-literal::

    array([<Axes: ylabel='Amplitude \n[MHz / $2\\pi$]'>, <Axes: xlabel='Time [ns]', ylabel='Population'>], dtype=object)




.. image:: 02A_Single_qubit_state_preparation_files/02A_Single_qubit_state_preparation_19_1.png


As expected, we get a partial transfer and a low fidelity.

.. code:: ipython3

    print(f"State fidelity: {zeroone.get_value(times)}")


.. parsed-literal::

    State fidelity: 0.3473607127051165


Optimization
------------

We define an optimizer and link our fidelity measure as a goal function
and the parameters of the cosine tone.

.. code:: ipython3

    optmap = OptimizationMap()
    optmap.add(gen, [params_gen[0], params_gen[2]])
    opt = ScipyOptimizerGradient(measure_and_gradient_func=zeroone.get_value_and_gradient, optimization_map=optmap)

One might think that the gradient associated with ``qubit_hamiltonian``
would not be empty, but instead

.. code:: ipython3

    qubit_hamiltonian.get_value_and_gradient(np.array([t_simu]))




.. parsed-literal::

    (Array([[[-1.508e+10, -2.493e+08],
             [-2.493e+08,  1.508e+10]]], dtype=float64),
     Array([], shape=(1, 0, 2, 2), dtype=float64))



This is because the parameters have been passed, but not “registered” by
``optmap``. To remedy this

.. code:: ipython3

    optmap.register_params_with_optimizables()
    print(qubit_hamiltonian.get_value_and_gradient(np.array([t_simu])))


.. parsed-literal::

    (Array([[[-1.508e+10, -2.493e+08],
            [-2.493e+08,  1.508e+10]]], dtype=float64), Array([[[[ 0.   , -0.992],
             [-0.992,  0.   ]],
    
            [[ 0.   , -0.315],
             [-0.315,  0.   ]]]], dtype=float64))


.. code:: ipython3

    envelope.get_value_and_gradient(jnp.array([t_simu]))




.. parsed-literal::

    (Array(2.513e+08, dtype=float64), Array([[1.]], dtype=float64))



In this case, as expected, we get two matrices which represent the
gradient of the qubit Hamiltonian with respect to the two parameters we
want to optimize, which can be accessed via the optmap as

.. code:: ipython3

    print(optmap.get_all_parameters())


.. parsed-literal::

    [Amplitude: 40 MHz x 2pi, lo_freq: 4.85 GHz x 2pi]


We can now run the optimization as

.. code:: ipython3

    print(opt.optimize(times))


.. parsed-literal::

    {'status': 1, 'value': 5.020428517354958e-13, 'iterations': 7, 'evaluations': 11, 'message': 'CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH'}


The new optimal parameters are

.. code:: ipython3

    print(optmap.get_all_parameters())


.. parsed-literal::

    [Amplitude: 50.2 MHz x 2pi, lo_freq: 4.8 GHz x 2pi]


We can now plot the optimized dynamics

.. code:: ipython3

    plot_signal_and_dynamics(gen, prop, ts, state_labels=[r"$|0\rangle$", r"$|1\rangle$"])




.. parsed-literal::

    array([<Axes: ylabel='Amplitude \n[MHz / $2\\pi$]'>, <Axes: xlabel='Time [ns]', ylabel='Population'>], dtype=object)




.. image:: 02A_Single_qubit_state_preparation_files/02A_Single_qubit_state_preparation_36_1.png


We can see from the plot and optimizer output that we have found good
controls.

.. code:: ipython3

    print(f"State fidelity: {zeroone.get_value(times)}")


.. parsed-literal::

    State fidelity: 0.999999999999498


In this notebook, we focused on state preparation. In the next notebook,
we will show how to perform the optimization directly in terms of gate
fidelity.
