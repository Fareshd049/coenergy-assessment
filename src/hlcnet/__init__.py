"""hlcnet: multi-indicator prediction of building thermal performance from short tests.

From 8 h of indoor temperature ``Tint(t)`` and heating power ``U(t)`` (one step
per minute), predict ``[HLC, HLC_ext, HLC_mit, tau_b, tau_n, tau_inf]`` with a
calibrated uncertainty for each, under the physical constraints
``HLC = HLC_ext + HLC_mit`` and ``0 < tau_n <= tau_inf <= tau_b``.

Modules, in the order of the report (``notebooks/report.ipynb``):

* :mod:`hlcnet.targets` - canonical order of the 6 indicators, families, units.
* :mod:`hlcnet.data` - memory-safe loading, file-level split, normalization, sensor noise.
* :mod:`hlcnet.eda` - full-dataset scan and exploratory figures.
* :mod:`hlcnet.identifiability` - natural experiments, ridge, PCA, twins and Delta test.
* :mod:`hlcnet.models` - CNN-GRU and variants, physics head, uncertainty head.
* :mod:`hlcnet.losses` - Gaussian beta-NLL and MSE, per-family weighting.
* :mod:`hlcnet.train` - training loop, early stopping, checkpoints.
* :mod:`hlcnet.metrics` - per-indicator and per-family metrics, constraints, coverage.
* :mod:`hlcnet.analysis` - calibration, log-normal bias, noise, sub-groups, ensembles.
* :mod:`hlcnet.specialist` - ridge and MLP specialists of one excitation.
* :mod:`hlcnet.excitation_design` - excitation-by-excitation evaluation, predicted-sigma criterion.
* :mod:`hlcnet.rocket` - ROCKET baseline.
* :mod:`hlcnet.report` - read saved results for the notebook (no retraining).
* :mod:`hlcnet.config`, :mod:`hlcnet.utils` - YAML configuration, seeds, device.
* :mod:`hlcnet.util_assessement` - loader provided by coEnergy, kept unchanged.
"""
