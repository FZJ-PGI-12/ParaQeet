.. _concepts:

Concepts and architecture
=========================

ParaQeet is organized as a stack of layers. Each layer interacts only with the
layer before it in the hierarchy, which keeps the codebase modular: you can swap
a pulse parametrization, a propagator, or an optimizer without touching the
rest of the setup. The codebase follows the modular structure shown below:

.. image:: ./layers.svg
   :align: center
   :width: 100%
   :alt: The ParaQeet layer stack

The modules
-----------

**Signal** (:mod:`paraqeet.signal`)
   Pulse parametrizations. Envelope shapes such as
   :class:`~paraqeet.signal.envelopes.GaussEnvelope` or
   :class:`~paraqeet.signal.envelopes.DCRABEnvelope` are combined by generators
   like the :class:`~paraqeet.signal.iq_mixer.IQMixer` (envelope mixed with a
   local oscillator) or the :class:`~paraqeet.signal.pwc_generator.PWCGenerator`
   (piecewise-constant bins for GRAPE) into the control signal seen by the
   system.

**Hamiltonian** (:mod:`paraqeet.hamiltonian`)
   The physical system: Hamiltonians for qubits, transmons, and resonators,
   couplings and drives, composed with
   :class:`~paraqeet.hamiltonian.composite_hamiltonian.CompositeHamiltonian`.
   Define your own Hamiltonian by following the :class:`~paraqeet.hamiltonian.hamiltonian.Hamiltonian`
   class structure.
   
**Equation of motion (EOM)** (:mod:`paraqeet.eom`)
   The eom layer turns a Hamiltonian into an equation of motion — the
   :class:`~paraqeet.eom.schroedinger_equation.SchroedingerEquation` for
   closed systems or the Lindblad
   :class:`~paraqeet.eom.master_equation.MasterEquation` for open systems.

**Propagation** (:mod:`paraqeet.propagation`)
   Solvers of the equation of motion, from piecewise matrix exponentials
   (:class:`~paraqeet.propagation.expm.Expm`) to Euler, Runge-Kutta and Verner ODE
   integrators. Gradients come from wrapping any of them in
   :class:`~paraqeet.propagation.goat.GOAT`, :class:`~paraqeet.propagation.grape.GRAPE` or
   :class:`~paraqeet.propagation.auto_diff_gradients.AutoDiffGradients`, which add analytic
   or automatically differentiated gradients to the propagation they wrap.

**Measurement** (:mod:`paraqeet.measurement`)
   Goal functions: state transfer and unitary fidelities, the Makhlin
   functional, pulse smoothness, and weighted sums of goals. A measurement
   reduces a propagated state to the scalar that the optimizer minimizes.

**Optimizers** (:mod:`paraqeet.optimizers`)
   Gradient-based (:class:`~paraqeet.optimizers.scipy_optimizer_gradient.ScipyOptimizerGradient`)
   and gradient-free (CMA-ES, Bayesian) algorithms, including a gradient-based dCRAB optimizer. 

Parameters: ``Quantity`` and ``OptimizationMap``
------------------------------------------------

Fundamental classes that connect the various modules across the package:

- A :class:`~paraqeet.quantity.Quantity` represents every tunable value —
  amplitude, frequency, coupling strength — together with its bounds and unit.
  Internally it is stored on a normalized scale, so optimizers always work on
  well-conditioned values regardless of physical magnitude. Quantities can be
  derived from other quantities via relations and update automatically.

- An :class:`~paraqeet.optimizable.Optimizable` class that represents classes that
  contain parameters that can be optimized. In case an
  :class:`~paraqeet.optimizable.Optimizable` class is also 
  :class:`~paraqeet.differentiable.Differentiable` it also provides gradients with 
  repspect to its own parameters.

- An :class:`~paraqeet.optimization_map.OptimizationMap` collects which
  quantities from :class:`~paraqeet.optimizable.Optimizable` classes that 
  are optimized in a given run. This makes the choice of optimization variables
  explicit and independent of the model definition: the same setup can optimize 
  two parameters or twenty.

Choosing an optimization method
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 34 20 28

   * - Method
     - Pulse parametrization
     - Gradients
     - Example
   * - GOAT
     - Analytic envelopes (Gaussian, flat-top, …)
     - Analytic, exact
     - :doc:`notebooks/02B_Single_qubit_gate`
   * - GRAPE
     - Piecewise-constant bins
     - Analytic, exact
     - :doc:`notebooks/04A_GRAPE_TLS`
   * - GOAToverGRAPE
     - Analytic envelopes, propagated piecewise
     - Chain rule of GOAT through GRAPE
     - :doc:`notebooks/04C_Single_qubit_gate_GOAToverGRAPE`
   * - Gradient-free
     - Any
     - None (CMA-ES, Bayesian)
     - :doc:`notebooks/02C_Qubit-bayesian-optimization`

As a rule of thumb: use GOAT when a few physical pulse parameters should stay
interpretable, GRAPE when you want maximum pulse flexibility per time bin, and
GOAToverGRAPE when you want smooth analytic pulses with the propagation
efficiency of GRAPE. Gradient-free methods are a fallback for measures without
gradients, e.g. when optimizing directly against an experiment.

Refer to the :doc:`example gallery <notebooks/index>` for examples on all of the above.
