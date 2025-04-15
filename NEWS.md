# tabnet 0.7.0

## Bugfixes

* Remove long-run example raising a Note.
* fix `tabet_pretrain` failing with `value_error("Can't convert data of class: 'NULL'")` in R 4.5
* fix `tabet_pretrain` wrongly used instead of `tabnet_fit` in Missing data predictor vignette
* improve message related to case_weights not being used as predictors.
* improve function documentation consistency before translation.
* fix "..." is not an exported object from 'namespace:dials'" error when using tune() on tabnet parameters. (#160 @cphaarmeyer)

# tabnet 0.6.0

## New features

* parsnip models now allow transparently passing case weights through `workflows::add_case_weights()` parameters  (#151)
* parsnip models now support `tabnet_model` and `from_epoch` parameters  (#143)

## Bugfixes

*  Adapt `tune::finalize_workflow()` test to {parsnip} v1.2 breaking change. (#155)
*  `autoplot()` now position the "has_checkpoint" points correctly when a `tabnet_fit()` is continuing a previous training using `tabnet_model =`. (#150)
*  Explicitely warn that `tabnet_model` option will not be used in `tabnet_pretrain()` tasks. (#150)

# tabnet 0.5.0

## New features

* {tabnet} now allows hierarchical multi-label classification through {data.tree} hierarchical `Node` dataset.  (#126) 
* `tabnet_pretrain()` now allows different GLU blocks in GLU layers in encoder and in decoder through the `config()` parameters `num_idependant_decoder` and `num_shared_decoder` (#129) 
* Add `reduce_on_plateau` as option for `lr_scheduler` at `tabnet_config()` (@SvenVw, #120)
* use zeallot internally with %<-% for code readability (#133)
* add FR translation (#131)

# tabnet 0.4.0

## New features

* Add explicit legend in `autoplot.tabnet_fit()` (#67)
* Improve unsupervised vignette content. (#67)
* `tabnet_pretrain()` now allows missing values in predictors. (#68)
* `tabnet_explain()` now works for `tabnet_pretrain` models. (#68)
* Allow missing-values values in predictor for unsupervised training. (#68)
* Improve performance of `random_obfuscator()` torch_nn module. (#68)
* Add support for early stopping (#69)
* `tabnet_fit()` and `predict()` now allow **missing values** in predictors. (#76)
* `tabnet_config()` now supports a `num_workers=` parameters to control parallel dataloading (#83)
* Add a vignette on missing data (#83)
* `tabnet_config()` now has a flag `skip_importance` to skip calculating feature importance (@egillax, #91)
* Export and document `tabnet_nn`
* Added `min_grid.tabnet` method for `tune` (@cphaarmeyer, #107)
* Added `tabnet_explain()` method for parsnip models (@cphaarmeyer, #108)
* `tabnet_fit()` and `predict()` now allow **multi-outcome**, all numeric or all factors but not mixed. (#118)

## Bugfixes

* `tabnet_explain()` is now correctly handling missing values in predictors. (#77)
* `dataloader` can now use `num_workers>0` (#83)
* new default values for `batch_size` and `virtual_batch_size` improves performance on mid-range devices.
* add default `engine="torch"` to tabnet parsnip model (#114)
* fix `autoplot()` warnings turned into errors with {ggplot2} v3.4 (#113)


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
