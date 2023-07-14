<<<<<<< HEAD
# make_tabular_data
Inspired by `sklearn.datasets.make_classification`, which in turn is based on work for NIPS 2003 feature selection challenge [1] - targeting linear classifiers.  Here the focus is on generating more complex, nonlinear datasets appropriate for use with deep learning/black box models which 'need' nonlinearity - otherwise you would/should use a simpler model.
=======
# synthetic-data
Inspired by `sklearn.datasets.make_classification`, which in turn is based on work for the NIPS 2003 feature selection challenge [1] - targeting linear classifiers.  Here the focus is on generating more complex, nonlinear datasets appropriate for use with deep learning/black box models which 'need' nonlinearity - otherwise you would/should use a simpler model.
>>>>>>> 329e54a (Updating readme)


### Approach
Ideally, the method would provide a concise specification to generate tabular data with sensible defaults.  The specification should provide `knobs` that the end user can dial up or down to see it's downstream impact.

Copulas are a model for specifying
the joint probability p(x1, x2, ..., xn) given a correlation structure along
with specifications for the marginal distribution of each feature. The current implementation uses a multivariate normal distribution with specified covariance matrix.  Future work can expand this choice to other multivariate distributions. 

<<<<<<< HEAD
=======
### Features:
Inputs:
- [x] specify the marginal distribution of a column  
- [x] correlation (correlated but not dependent)
- [x] prescaled inputs using MinMaxScaler (TODO:  add StdScaler, etc.)
- [ ] nuisance variables - carry no signal, set the lower limit on 'acceptable' feature importance
- [ ] redundant (correlated and dependent - say by a linear combo of informative features)

<<<<<<< HEAD
<<<<<<< HEAD
- [ ] separation between classes (can we filter +/- k% on either side of p_thresh to create separation?)
- [ ] overlap - since we have ground truth probabilities, we could sample from a binomial distribution with probability of (py|x) to determine labels - this would work in conjuction with sig_k which controls the steepness of the sigmoid  
- [ ] noise level - apply *after* we generate regression values/labels 
  - [x] gaussian white on X
  - [ ] percentage shuffled  (see e.g. `flip_y` in `make_classification`)
- [x] categorical features (stretch)
- [ ] outlier generation (stretch, *new*)
- [ ] create fake PII with [pydbgen](https://github.com/tirthajyoti/pydbgen)  (stretch, *new*)


Output:
- [ ] functional dependence y_reg = f(x) where y_reg is a float - implemented via an expression as a string
- [ ] mapping from y_reg value to y_class 
    - [ ] partition and label - e.g. `y_class = y_reg < np.median(y_reg)`
    - [x] sigmoid
    - [ ] Gompertz curve (a parameterized sigmoid - would give control over uncertainty?
    - these last two provide ground truth P(y|x) (regression -> probability -> label)
  -[ ] noise (e.g. `flip_y`)
  -[ ] map class to probability using random draw from binomial distribution

>>>>>>> b75f94d (cleaned up comments and removed sympy from req)
### Parameters  
=======
### Parameters
>>>>>>> 411e66f (whitespace hook added)
=======
### Parameters  
>>>>>>> a230a51 (Revert "whitespace hook added")
| name          | type       | default        | description                                                                                                                      |
| ------------- | ---------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| n_samples     | int        | (default=100)  | The number of samples.                                                                                                           |
| n_informative | int        | (default=2)    | The number of informative features - these should all be represented in the symbolic expression used to generate y_reg           |
| n_nuisance    | int        | (default=0)    | The number of nuisance features - these *should not* be included in the symbolic expression - and hence have no role in the DGP. |
| n_clases      | int        | (default=2)    | the number of classes                                                                                                            |
| dist          | list       |                | a list of the marginal distributions to apply to the features/columns                                                            |
| cov           | matrix     |                | a square numpy array with dimensions (??? x ???) - should be n_total where n_total=n_informative + n_nuisance                    |
| expr          | str |                | an expression providing y = f(X)                                                                                                 |
| sig_k         | float      | (default=1.0)  | the steepness of the sigmoid used in mapping y_reg to y_prob                                                                     |
| sig_x0        | float      | (default=None) | the center point of the sigmoid used in mappying y_reg to y_prob                                                                 |
| p_thresh      | float      | (default=0.5)  | threshold probability that determines boundary between classes                                                                   |
| noise_level_x | float      | (default=0.0)  | level of Gaussian white noise to apply to X                                                                                      |
| noise_level_y | float      | (default=0.0)  | level of Gaussian white noise to apply to y_label (like `flip_y`)                                                                |


## Getting Started

### Local Installation
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Tests
<<<<<<< HEAD
To run tests:
```bash
<<<<<<< HEAD
$ python -m pytest tests/
=======
$ pip install -r requirements-test.txt
$ make test_local
>>>>>>> 8533f8d (reformatted the continuous attributes dictionary, fixed tests)
=======

Test/Lint Dependencies

```bash
pip install -r requirements-test.txt
```

To run tests:
```bash
make test_local
>>>>>>> 4850707 (removed unused imports)
```

#### Pre-Commit
To install `pre-commit` hooks, run the following commands:

```cli
pre-commit install
pre-commit run
```

### Referencing this library
If you use this library in your work, please cite our paper:  
```
@inproceedings{barr:2020,
  author    = {Brian Barr and Ke Xu and Claudio Silva and Enrico Bertini and Robert Reilly and  C. Bayan Bruss and Jason D. Wittenbach},
  title     = {{Towards Ground Truth Explainability on Tabular Data}},
  year      = {2020},
  maintitle = {International Conference on Machine Learning},
  booktitle = {2020 ICML Workshop on Human Interpretability in Machine Learning (WHI 2020)},
  date = {2020-07-17},
  pages = {362-367},
}                             
```

### Notes
If you have tabular data, and want to fit a copula from it, consider this python library:  [copulas](https://sdv-dev.github.io/Copulas/index.html)  
Quick [visual tutorial](https://twiecki.io/blog/2018/05/03/copulas/) of copulas and probability integral transform.

To run the examples, you should run:
```bash
$ python -m pip install pandas pytest pytest-cov seaborn shap tensorflow "DataProfiler[full]"
```

### References
[1] Guyon, “Design of experiments for the NIPS 2003 variable selection benchmark”, 2003.
