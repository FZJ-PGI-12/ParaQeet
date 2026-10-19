<div align="center">
  <center><img src="https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/-/raw/main/big_logo.png" alt="paraqeet_logo" width="90%"/></center>
</div>

# ParaQeet -  A quantum optimal control toolkit with simple parameter management

[![pipeline status](https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/badges/main/pipeline.svg)](https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/-/commits/main)
[![coverage report](https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/badges/main/coverage.svg)](https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/-/commits/main)
[![Latest Release](https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/-/badges/release.svg)](https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/-/releases)
[![readthedocs](https://readthedocs.org/projects/paraqeet/badge/?version=latest)](https://paraqeet.readthedocs.io/en/latest/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI - Version](https://img.shields.io/pypi/v/paraqeet)](https://pypi.org/project/paraqeet/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/paraqeet.svg)]((https://pypi.org/project/paraqeet/))
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23208578.svg)](https://doi.org/10.5281/zenodo.23208578)

*Note: This is a preview version, a 1.0.0 release is forthcoming.*

>[!note]
> Development happens on [JuGit](https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet). The [GitHub repository](https://github.com/FZJ-PGI-12/paraqeet) is a read-only mirror, so please open issues and merge requests on JuGit.


Choose a pulse parametrisation, simulate a quantum system, and optimize. 

Combining Quantum Optimal Control (QOC) methods with automatic differentiation with JAX.
Aimed at resource efficient computation.

ParaQeet is organized as a stack of layers. Each layer interacts only with the
layer before it in the hierarchy, which keeps the codebase modular: you can swap
a pulse parametrization, a propagator, or an optimizer without touching the
rest of the setup. The codebase follows the modular structure shown below:

<div align="center">
  <center><img src="https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/-/raw/main/docs/layers.svg" alt="Layers" width="90%"/></center>
</div>

Currently implemented QOC methods:
- **GRAPE**: Gradient Ascent Pulse Engineering
- **GOAT**: Gradient Optimization of Analytic conTrols
- **dCRAB** : (Gradient based) dressed Chopped RAndom Basis
- **GOAToverGRAPE**: A variant of GROUP that optimizes continuous pulse parameters with GRAPE inside.
- **AD**: Automatic differentiation of the state/propagator evolution

Currently implemented propagation methods:
- **Expm**: Matrix exponential using JAX ``expm``
- **ExpmChebyshev**: Matrix exponential using Chebyshev polynomial expansion
- **ODE solvers**: Diffrax, Verner 7th order method, and Scipy Runge-Kutta methods

Propagation methods can be freely combined with QOC methods, e.g. GOAT optimization using the ``ExpmChebyshev`` propagator.


## Installation

### From PyPI

Install the latest release with:

```bash
pip install paraqeet
```

### From source

To install the current `main` branch for development, follow the instructions in [CONTRIBUTING.md](CONTRIBUTING.md).


## Contributing

We warmly welcome all contributions and encourage contributors to keep the codebase semantically clear and readable. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions.

We recommend that contributors verify and test their code themselves, and declare any AI usage in merge requests. We also recommend that contributors create commits themselves, rather than relying on AI.


## Citation

If you use this software, please cite it. Author list, ORCIDs, and the associated references are maintained in [`CITATION.cff`](CITATION.cff).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23208578.svg)](https://doi.org/10.5281/zenodo.23208578)

The badge above always resolves to the latest archived release. To cite a specific version instead, for example for reproducibility, use that version's own DOI.

## License

Apache 2.0. See [LICENSE](LICENSE). Copyright &copy; 2026 Forschungszentum Jülich GmbH.


## Authors
Developed at the Institute for Quantum Computing Analytics (PGI-12), Forschungszentrum Jülich GmbH, by:

- Ashutosh Mishra 
- Nicolas Wittler 
- Alessandro Ciani 
- Lidia Westphal 
- Alexander Simm 
- Moritz Wald 


## Acknowledgments

The codebase contains code thoroughly written and verified by the authors. Some parts were made with the assistance of AI models, including Claude Opus (Anthropic) and open-weight models such as DeepSeek.

