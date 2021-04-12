# tabnet (development version)

* Fixed bug in GPU training. (#22)
* Added GPU CI. (#22)
* Explicit error in case of missing values (@cregouby, #24)
* Allow model fine-tuning through passing a pre-trained model to tabnet_fit() (@cregouby, #26)
* Fixed memory leaks when using custom autograd function.
* Better handling of larger datasets when running `tabnet_explain`.
* Batch predictions to avoid OOM error.
* Add tabnet_pretrain() for unsupervised pretraining (@cregouby, #29)
* Add autoplot() of model loss among epochs (@cregouby, #36)
* Added a `config` argument to fit/pretrain functions so one can pass a pre-made config list. (#42)
* Add `mask_type` configuration option with `entmax` additional to `sparsemax` (@cmcmaster1, #48)

# tabnet 0.1.0

* Added a `NEWS.md` file to track changes to the package.
