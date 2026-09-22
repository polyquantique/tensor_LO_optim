# tensor_LO_optim

Code accompanying the paper **"A tensor framework for optimal local oscillators in the homodyne detection of multiphoton-number states"**

## Overview

This repository provides a tensor-based framework for finding optimal local oscillator (LO) modes for homodyne detection of multiphoton number states. It includes tools for:

- Constructing and decomposing joint spectral amplitude (JSA) tensors for Gaussian and higher-order parametric downconversion (HOPDC) sources using randomized SVD / HOSVD.
- Deriving and numerically testing eigenvalue bounds and conditions relevant to these JSA tensors.
- Comparing the fidelity between the optimal local oscillator and the leading HOSVD mode.
- Optimizing local oscillator modes via a gradient-descent scheme with a hybrid penalty and homotopy continuation method.

## Installation

```bash
git clone https://github.com/polyquantique/tensor_LO_optim.git
cd tensor_lo_optim
pip install -e .
```

### Dependencies

```
numpy
scipy
matplotlib
jax
jaxopt
pandas
seaborn
tensorrsvd @ git+https://github.com/PaulVirally/TensorRSVD
```

These are listed in `pyproject.toml` and will be installed automatically with `pip install -e .`. 

## Repository structure

### `src/optimization_core/` — main library

The core package implementing the optimization:

- `gaussian_jsa.py`:  construction of Gaussian JSA tensors.
- `waveguide_spdc_jsa.py`: construction of JSA tensors for higher-order parametric downconversion (HOPDC) in waveguides.
- `objectives.py`: objective functions used to optimize the local oscillator mode.
- `optimizers.py`: optimization routines, including the penalty routine, the hybrid penalty and homotopy continuation routine with gradient descent, and trust-region methods.
- `run_optimization.py`: functions to launch an optimization run end to end using the optimizers previously defined.
- `utils.py`: helper for loading and saving data, and complex vectors handling.
- `bounds.py`: functions to calculate the homodyne visibility upper bounds.
- Additional modules with complementary plotting functions for visualizing JSAs, modes, and optimization results.

### `optimization_experiments/` — examples and paper figures

- **`0_example_usage_random_svd/`**
  Example usage of randomized SVD applied to Gaussian JSAs and to JSAs associated with higher-order parametric downconversion (HOPDC).

- **`1_mode_decompositions/`**
  Mode decompositions of Gaussian and HOPDC JSAs using randomized SVD.

- **`2_Motivating example_initialization/`**
  Example illustrating the motivation behind the initialization strategy used in the optimization routines.

 - **`3_1_tensor_eigenvalues_gaussians/`**  Calculation of bounds and tests of the eigenvalue condition for Gaussian JSA tensors.
 - **`3_1_tensor_eigenvalues_hopdc/`**   Calculation of bounds and tests of the eigenvalue condition for HOPDC JSA tensors.
  - **`3_1_tensor_eigenvalues_hopdc_chirped_pump/`**  Calculation of bounds and tests of the eigenvalue condition for HOPDC JSA tensors with a chirped pump.

- **`4_optimal_mode_distributions_fidelity/`**
  Calculation of the fidelity between the optimal local oscillator and the leading HOSVD mode.

- **`5_computation/`**
  Examples of the computation of optimal local oscillators for homodyne detection of Gaussian and HOPDC JSAs:
  - Optimization using a hybrid penalty method and homotopy continuation on a gradient-descent method.
  - Examples/comparison with a trust-region method.
  - Example of an optimization/continuation schedule.

  ## Citation

If you use this code, please cite the associated paper:

```bibtex
@article{tensor_lo_optim,
  title   = {A tensor framework for optimal local oscillators in the homodyne detection of multiphoton-number states},
  author  = {TBD},
  journal = {TBD},
  year    = {TBD},
  note    = {Manuscript in preparation}
}
```

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
