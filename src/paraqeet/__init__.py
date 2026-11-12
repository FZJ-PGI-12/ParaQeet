"""**ParaQeet: A quantum optimal control toolkit with simple parameter management.**

Choose a pulse parametrization, simulate a quantum system, and optimize.
The package is organized in layers, each interacting only with the layer
above it in the hierarchy:

- ``signal``: pulse parametrizations (envelopes, generators, mixers).
- ``hamiltonian``: Hamiltonian, composite Hamiltonian, and drive Hamiltonians.
- ``eom``: different equations of motion.
- ``propagation``: solvers of the equation of motion.
- ``measurement``: fidelities and other goal functions.
- ``optimizers``: optimization algorithms (gradient based and gradient free).

All tunable values are represented by ``Quantity`` objects. The parameters to
optimize are collected in an ``OptimizationMap``, which is handed to an
optimizer together with a goal function.

The package borrows ideas of semi-automatic differentiation from :cite:p:`goerz2022quantum`,
and combines automatic differentiation with analytic quantum optimal control gradients.

"""

import jax

from paraqeet.eom.master_equation import MasterEquation
from paraqeet.eom.schroedinger_equation import SchroedingerEquation
from paraqeet.measurement.fidelity import Fidelity, FidelityGRAPE, UnitaryFidelity
from paraqeet.optimization_map import OptimizationMap
from paraqeet.optimizers.scipy_optimizer import ScipyOptimizer, ScipyOptimizerGradient
from paraqeet.propagation.expm import Expm
from paraqeet.propagation.goat import GOAT
from paraqeet.propagation.grape import GRAPE
from paraqeet.propagation.vern7 import Vern7
from paraqeet.quantity import Array, Quantity

jax.config.update("jax_enable_x64", True)

# aliases
OptMap = OptimizationMap
SchrEq = SchroedingerEquation


__all__ = [
    "OptimizationMap",
    "OptMap",
    "Array",
    "Quantity",
    "SchroedingerEquation",
    "SchrEq",
    "MasterEquation",
    "Expm",
    "Fidelity",
    "FidelityGRAPE",
    "Vern7",
    "GOAT",
    "GRAPE",
    "UnitaryFidelity",
    "ScipyOptimizer",
    "ScipyOptimizerGradient",
]
