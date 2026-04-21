# PROMOTED

GPU scaled sparse RKAN training with dense-vs-P4 update substrates.

Samples: 30
Feature count: 1000000
Steps per sample: 1000
Batch size: 65536
Observed training rows per substrate/sample: 65536000
Rationalized coefficients: ['1/1', '1/1', '-1/6']
Identity passed: True
Convergence passed: True
Dense/P4 training speedup mean 3.72578 [3.68015, 3.76787]
Dense/P4 total speedup mean 3.71753 [3.6729, 3.75924]
P4/Dense memory ratio mean 0.41773 [0.41773, 0.41773]
Symbolic expression: `(6*x_0**2 - x_1**3 + 6*x_1)/6`
PDF compiled: True
