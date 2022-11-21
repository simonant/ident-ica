## Function Classes for Identifiable Nonlinear ICA
This repository provides the code to reproduce the experiments from
Appendix H of the paper 'Function Classes for Identifiable Nonlinear Independent Component Analysis' 
(https://arxiv.org/abs/2208.06406). These results shall illustrate the theorems (in particular Theorem 4) from the paper.
Note that training the regularized model for the rotation function is slightly unstable and sometimes
diverges.
For an in depth experimental evaluation the training stability needs to be improved and/or 
outliers have to be removed from the statistical analysis.

### Installation

To reproduce the experiments clone the repository and then run:

```
pip install -r requirements.txt
python -m run_experiments.exp
```