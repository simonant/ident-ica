## Function Classes for Identifiable Nonlinear ICA
This repository provides the code to reproduce the experiments from
Appendix H of the paper 'Function Classes for Identifiable Nonlinear Independent Component Analysis' 
(https://arxiv.org/abs/2208.06406). These results shall illustrate the theorems from the paper.
Note that in few cases the training of the regularized model diverges for the experiment with
the rotation function. We used a seed that had no such case present.
Otherwise outliers have to be removed from the statistical analysis.

### Installation

To reproduce the experiments clone the repository and then run:

```
pip install -r requirements.txt
python -m run_experiments.exp
```