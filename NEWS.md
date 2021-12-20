# tabnet (development version)

## New features

* Add explicit legend in `autoplot.tabnet_fit()` (#67)
* Improve unsupervised vignette content. (#67)
* `tabnet_pretrain()` now allows missing values in predictors. (#68)
* `tabnet_explain()` now works for `tabnet_pretrain` models. (#68)
* Improve performance of `random_obfuscator()` torch_nn module. (#68)
* `tabnet_fit()` and `predict()` now allow missing values in predictors. (#76)

# tabnet 0.3.0

* Added an `update` method for tabnet models to allow the correct usage of `finalize_workflow` (#60).

# tabnet 0.2.0

## New features

* Allow model fine-tuning through passing a pre-trained model to `tabnet_fit()` (@cregouby, #26)
* Explicit error in case of missing values (@cregouby, #24)
* Better handling of larger datasets when running `tabnet_explain()`.
* Add `tabnet_pretrain()` for unsupervised pretraining (@cregouby, #29)
* Add `autoplot()` of model loss among epochs (@cregouby, #36)
* Added a `config` argument to `fit() / pretrain()` so one can pass a pre-made config list. (#42)
* In `tabnet_config()`, new `mask_type` option with `entmax` additional to default `sparsemax` (@cmcmaster1, #48)
* In `tabnet_config()`, `loss` now also takes function (@cregouby, #55)

## Bugfixes

* Fixed bug in GPU training. (#22)
* Fixed memory leaks when using custom autograd function.
* Batch predictions to avoid OOM error.

## Internal improvements

* Added GPU CI. (#22)

# tabnet 0.1.0

* Added a `NEWS.md` file to track changes to the package.
