# Changelog

## tabnet 0.8.0

CRAN release: 2026-01-31

### New features

- messaging is now improved with {cli}
- add optimal threshold and support size into new 1.5 alpha
  [`entmax15()`](../reference/entmax15.md) and
  [`sparsemax15()`](../reference/sparsemax.md) `mask_types`. Add an
  optional `mask_topk` config parameter.
  ([\#180](https://github.com/mlverse/tabnet/issues/180))
- `optimizer`now default to the `torch_ignite_adam` when available.
  Result is 30% faster pretraining and fitting tasks
  ([\#178](https://github.com/mlverse/tabnet/issues/178)).
- add [`nn_aum_loss()`](../reference/nn_aum_loss.md) function for area
  under the $Min(FPR,FNR)$ optimization for cases of unbalanced binary
  classification
  ([\#178](https://github.com/mlverse/tabnet/issues/178)).
- add a vignette on imbalanced binary classification with
  [`nn_aum_loss()`](../reference/nn_aum_loss.md)
  ([\#178](https://github.com/mlverse/tabnet/issues/178)).

### Bugfixes

- config parameter now merge correctly for torch loss or torch optimizer
  generator.
- `nn_unsupervised_loss()` is now a proper loss function.

## tabnet 0.7.0

CRAN release: 2025-04-16

### Bugfixes

- Remove long-run example raising a Note.
- fix `tabet_pretrain` failing with
  `value_error("Can't convert data of class: 'NULL'")` in R 4.5
- fix `tabet_pretrain` wrongly used instead of `tabnet_fit` in Missing
  data predictor vignette
- improve message related to case_weights not being used as predictors.
- improve function documentation consistency before translation.
- fix “…” is not an exported object from ‘namespace:dials’” error when
  using tune() on tabnet parameters.
  ([\#160](https://github.com/mlverse/tabnet/issues/160)
  [@cphaarmeyer](https://github.com/cphaarmeyer))

## tabnet 0.6.0

CRAN release: 2024-06-15

### New features

- parsnip models now allow transparently passing case weights through
  [`workflows::add_case_weights()`](https://workflows.tidymodels.org/reference/add_case_weights.html)
  parameters ([\#151](https://github.com/mlverse/tabnet/issues/151))
- parsnip models now support `tabnet_model` and `from_epoch` parameters
  ([\#143](https://github.com/mlverse/tabnet/issues/143))

### Bugfixes

- Adapt
  [`tune::finalize_workflow()`](https://tune.tidymodels.org/reference/finalize_model.html)
  test to {parsnip} v1.2 breaking change.
  ([\#155](https://github.com/mlverse/tabnet/issues/155))
- [`autoplot()`](https://ggplot2.tidyverse.org/reference/autoplot.html)
  now position the “has_checkpoint” points correctly when a
  [`tabnet_fit()`](../reference/tabnet_fit.md) is continuing a previous
  training using `tabnet_model =`.
  ([\#150](https://github.com/mlverse/tabnet/issues/150))
- Explicitely warn that `tabnet_model` option will not be used in
  [`tabnet_pretrain()`](../reference/tabnet_pretrain.md) tasks.
  ([\#150](https://github.com/mlverse/tabnet/issues/150))

## tabnet 0.5.0

CRAN release: 2023-12-05

### New features

- {tabnet} now allows hierarchical multi-label classification through
  {data.tree} hierarchical `Node` dataset.
  ([\#126](https://github.com/mlverse/tabnet/issues/126))
- [`tabnet_pretrain()`](../reference/tabnet_pretrain.md) now allows
  different GLU blocks in GLU layers in encoder and in decoder through
  the `config()` parameters `num_idependant_decoder` and
  `num_shared_decoder`
  ([\#129](https://github.com/mlverse/tabnet/issues/129))
- Add `reduce_on_plateau` as option for `lr_scheduler` at
  [`tabnet_config()`](../reference/tabnet_config.md)
  ([@SvenVw](https://github.com/SvenVw),
  [\#120](https://github.com/mlverse/tabnet/issues/120))
- use zeallot internally with %\<-% for code readability
  ([\#133](https://github.com/mlverse/tabnet/issues/133))
- add FR translation
  ([\#131](https://github.com/mlverse/tabnet/issues/131))

## tabnet 0.4.0

CRAN release: 2023-05-11

### New features

- Add explicit legend in
  [`autoplot.tabnet_fit()`](../reference/autoplot.tabnet_fit.md)
  ([\#67](https://github.com/mlverse/tabnet/issues/67))
- Improve unsupervised vignette content.
  ([\#67](https://github.com/mlverse/tabnet/issues/67))
- [`tabnet_pretrain()`](../reference/tabnet_pretrain.md) now allows
  missing values in predictors.
  ([\#68](https://github.com/mlverse/tabnet/issues/68))
- [`tabnet_explain()`](../reference/tabnet_explain.md) now works for
  `tabnet_pretrain` models.
  ([\#68](https://github.com/mlverse/tabnet/issues/68))
- Allow missing-values values in predictor for unsupervised training.
  ([\#68](https://github.com/mlverse/tabnet/issues/68))
- Improve performance of `random_obfuscator()` torch_nn module.
  ([\#68](https://github.com/mlverse/tabnet/issues/68))
- Add support for early stopping
  ([\#69](https://github.com/mlverse/tabnet/issues/69))
- [`tabnet_fit()`](../reference/tabnet_fit.md) and
  [`predict()`](https://rdrr.io/r/stats/predict.html) now allow
  **missing values** in predictors.
  ([\#76](https://github.com/mlverse/tabnet/issues/76))
- [`tabnet_config()`](../reference/tabnet_config.md) now supports a
  `num_workers=` parameters to control parallel dataloading
  ([\#83](https://github.com/mlverse/tabnet/issues/83))
- Add a vignette on missing data
  ([\#83](https://github.com/mlverse/tabnet/issues/83))
- [`tabnet_config()`](../reference/tabnet_config.md) now has a flag
  `skip_importance` to skip calculating feature importance
  ([@egillax](https://github.com/egillax),
  [\#91](https://github.com/mlverse/tabnet/issues/91))
- Export and document `tabnet_nn`
- Added `min_grid.tabnet` method for `tune`
  ([@cphaarmeyer](https://github.com/cphaarmeyer),
  [\#107](https://github.com/mlverse/tabnet/issues/107))
- Added [`tabnet_explain()`](../reference/tabnet_explain.md) method for
  parsnip models ([@cphaarmeyer](https://github.com/cphaarmeyer),
  [\#108](https://github.com/mlverse/tabnet/issues/108))
- [`tabnet_fit()`](../reference/tabnet_fit.md) and
  [`predict()`](https://rdrr.io/r/stats/predict.html) now allow
  **multi-outcome**, all numeric or all factors but not mixed.
  ([\#118](https://github.com/mlverse/tabnet/issues/118))

### Bugfixes

- [`tabnet_explain()`](../reference/tabnet_explain.md) is now correctly
  handling missing values in predictors.
  ([\#77](https://github.com/mlverse/tabnet/issues/77))
- `dataloader` can now use `num_workers>0`
  ([\#83](https://github.com/mlverse/tabnet/issues/83))
- new default values for `batch_size` and `virtual_batch_size` improves
  performance on mid-range devices.
- add default `engine="torch"` to tabnet parsnip model
  ([\#114](https://github.com/mlverse/tabnet/issues/114))
- fix
  [`autoplot()`](https://ggplot2.tidyverse.org/reference/autoplot.html)
  warnings turned into errors with {ggplot2} v3.4
  ([\#113](https://github.com/mlverse/tabnet/issues/113))

## tabnet 0.3.0

CRAN release: 2021-10-11

- Added an `update` method for tabnet models to allow the correct usage
  of `finalize_workflow`
  ([\#60](https://github.com/mlverse/tabnet/issues/60)).

## tabnet 0.2.0

CRAN release: 2021-06-22

### New features

- Allow model fine-tuning through passing a pre-trained model to
  [`tabnet_fit()`](../reference/tabnet_fit.md)
  ([@cregouby](https://github.com/cregouby),
  [\#26](https://github.com/mlverse/tabnet/issues/26))
- Explicit error in case of missing values
  ([@cregouby](https://github.com/cregouby),
  [\#24](https://github.com/mlverse/tabnet/issues/24))
- Better handling of larger datasets when running
  [`tabnet_explain()`](../reference/tabnet_explain.md).
- Add [`tabnet_pretrain()`](../reference/tabnet_pretrain.md) for
  unsupervised pretraining ([@cregouby](https://github.com/cregouby),
  [\#29](https://github.com/mlverse/tabnet/issues/29))
- Add
  [`autoplot()`](https://ggplot2.tidyverse.org/reference/autoplot.html)
  of model loss among epochs ([@cregouby](https://github.com/cregouby),
  [\#36](https://github.com/mlverse/tabnet/issues/36))
- Added a `config` argument to `fit() / pretrain()` so one can pass a
  pre-made config list.
  ([\#42](https://github.com/mlverse/tabnet/issues/42))
- In [`tabnet_config()`](../reference/tabnet_config.md), new `mask_type`
  option with `entmax` additional to default `sparsemax`
  ([@cmcmaster1](https://github.com/cmcmaster1),
  [\#48](https://github.com/mlverse/tabnet/issues/48))
- In [`tabnet_config()`](../reference/tabnet_config.md), `loss` now also
  takes function ([@cregouby](https://github.com/cregouby),
  [\#55](https://github.com/mlverse/tabnet/issues/55))

### Bugfixes

- Fixed bug in GPU training.
  ([\#22](https://github.com/mlverse/tabnet/issues/22))
- Fixed memory leaks when using custom autograd function.
- Batch predictions to avoid OOM error.

### Internal improvements

- Added GPU CI. ([\#22](https://github.com/mlverse/tabnet/issues/22))

## tabnet 0.1.0

CRAN release: 2021-01-14

- Added a `NEWS.md` file to track changes to the package.
