# Robust Regression

This is a bunch of scripts to make produce the simulations for the paper, but organized in a handy package that can be expanded later in time.

## Code Organization

This package is subdivide into 5 other packages:
* `sweeps` : contains helpers to create sweeps of some parameters
* `aux_functions` : contains the definitions of fully vectorized (á la Numpy) function like $\mathcal{Z}_{\text{out}}$, $f_{\text{out}}$ and $f_{\mathbf{w}}$.
* `fixed_point_equations` : contains the definitions of the fixed point form for the problems studied. Also it contains the functions to run a single fixed point iteration or to optimize over some parameters.
* `regression_numerics` : contains the relevant routines to perform ERM simulations and AMP simulations
* `utils` : miscellaneus stuff used for numerical integration or root finding.

## Installation

To install the package in the virtual environment of your choice (`venv` or `condaenv`) first activate the virtual environment and then in the folder of this package run
```shell
$ pip install .
```

Once installed the package can be imported as

```python
import robust_regression as rr
```

## Contributing

Fell free to open pull request if you want to add something to the package.