# Parsnip compatible tabnet model

Parsnip compatible tabnet model

## Usage

``` r
tabnet(
  mode = "unknown",
  cat_emb_dim = NULL,
  decision_width = NULL,
  attention_width = NULL,
  num_steps = NULL,
  mask_type = NULL,
  mask_topk = NULL,
  num_independent = NULL,
  num_shared = NULL,
  num_independent_decoder = NULL,
  num_shared_decoder = NULL,
  penalty = NULL,
  feature_reusage = NULL,
  momentum = NULL,
  epochs = NULL,
  batch_size = NULL,
  virtual_batch_size = NULL,
  learn_rate = NULL,
  optimizer = NULL,
  loss = NULL,
  clip_value = NULL,
  drop_last = NULL,
  lr_scheduler = NULL,
  rate_decay = NULL,
  rate_step_size = NULL,
  checkpoint_epochs = NULL,
  verbose = NULL,
  importance_sample_size = NULL,
  early_stopping_monitor = NULL,
  early_stopping_tolerance = NULL,
  early_stopping_patience = NULL,
  skip_importance = NULL,
  tabnet_model = NULL,
  from_epoch = NULL
)
```

## Arguments

- mode:

  A single character string for the type of model. Possible values for
  this model are "unknown", "regression", or "classification".

- cat_emb_dim:

  Size of the embedding of categorical features. If int, all categorical
  features will have same embedding size, if list of int, every
  corresponding feature will have specific embedding size.

- decision_width:

  (int) Width of the decision prediction layer. Bigger values gives more
  capacity to the model with the risk of overfitting. Values typically
  range from 8 to 64.

- attention_width:

  (int) Width of the attention embedding for each mask. According to the
  paper n_d = n_a is usually a good choice. (default=8)

- num_steps:

  (int) Number of steps in the architecture (usually between 3 and 10)

- mask_type:

  (character) Final layer of feature selector in the
  attentive_transformer block, either `"sparsemax"`, `"entmax"` or
  `"entmax15"`.Defaults to `"sparsemax"`.

- mask_topk:

  (int) mask sparsity top-k for `sparsemax15` and `entmax15.` See
  [`entmax15()`](entmax15.md) for detail.

- num_independent:

  Number of independent Gated Linear Units layers at each step of the
  encoder. Usual values range from 1 to 5.

- num_shared:

  Number of shared Gated Linear Units at each step of the encoder. Usual
  values at each step of the decoder. range from 1 to 5

- num_independent_decoder:

  For pretraining, number of independent Gated Linear Units layers Usual
  values range from 1 to 5.

- num_shared_decoder:

  For pretraining, number of shared Gated Linear Units at each step of
  the decoder. Usual values range from 1 to 5.

- penalty:

  This is the extra sparsity loss coefficient as proposed in the
  original paper. The bigger this coefficient is, the sparser your model
  will be in terms of feature selection. Depending on the difficulty of
  your problem, reducing this value could help (default 1e-3).

- feature_reusage:

  (num) This is the coefficient for feature reusage in the masks. A
  value close to 1 will make mask selection least correlated between
  layers. Values range from 1 to 2.

- momentum:

  Momentum for batch normalization, typically ranges from 0.01 to 0.4
  (default=0.02)

- epochs:

  (int) Number of training epochs.

- batch_size:

  (int) Number of examples per batch, large batch sizes are recommended.
  (default: 1024^2)

- virtual_batch_size:

  (int) Size of the mini batches used for "Ghost Batch Normalization"
  (default=256^2)

- learn_rate:

  initial learning rate for the optimizer.

- optimizer:

  the optimization method. currently only `"adam"` is supported, you can
  also pass any torch optimizer function.

- loss:

  (character or function) Loss function for training (default to mse for
  regression and cross entropy for classification)

- clip_value:

  If a num is given this will clip the gradient at clip_value. Pass
  `NULL` to not clip.

- drop_last:

  (logical) Whether to drop last batch if not complete during training

- lr_scheduler:

  if `NULL`, no learning rate decay is used. If "step" decays the
  learning rate by `lr_decay` every `step_size` epochs. If
  "reduce_on_plateau" decays the learning rate by `lr_decay` when no
  improvement after `step_size` epochs. It can also be a
  [torch::lr_scheduler](https://torch.mlverse.org/docs/reference/lr_scheduler.html)
  function that only takes the optimizer as parameter. The `step` method
  is called once per epoch.

- rate_decay:

  multiplies the initial learning rate by `rate_decay` every
  `rate_step_size` epochs. Unused if `lr_scheduler` is a
  [`torch::lr_scheduler`](https://torch.mlverse.org/docs/reference/lr_scheduler.html)
  or `NULL`.

- rate_step_size:

  the learning rate scheduler step size. Unused if `lr_scheduler` is a
  [`torch::lr_scheduler`](https://torch.mlverse.org/docs/reference/lr_scheduler.html)
  or `NULL`.

- checkpoint_epochs:

  checkpoint model weights and architecture every `checkpoint_epochs`.
  (default is 10). This may cause large memory usage. Use `0` to disable
  checkpoints.

- verbose:

  (logical) Whether to print progress and loss values during training.

- importance_sample_size:

  sample of the dataset to compute importance metrics. If the dataset is
  larger than 1e5 obs we will use a sample of size 1e5 and display a
  warning.

- early_stopping_monitor:

  Metric to monitor for early_stopping. One of "valid_loss",
  "train_loss" or "auto" (defaults to "auto").

- early_stopping_tolerance:

  Minimum relative improvement to reset the patience counter. 0.01 for
  1% tolerance (default 0)

- early_stopping_patience:

  Number of epochs without improving until stopping training.
  (default=5)

- skip_importance:

  if feature importance calculation should be skipped (default: `FALSE`)

- tabnet_model:

  A previously fitted `tabnet_model` object to continue the fitting on.
  if `NULL` (the default) a brand new model is initialized.

- from_epoch:

  When a `tabnet_model` is provided, restore the network weights from a
  specific epoch. Default is last available checkpoint for restored
  model, or last epoch for in-memory model.

## Value

A TabNet `parsnip` instance. It can be used to fit tabnet models using
`parsnip` machinery.

## Threading

TabNet uses `torch` as its backend for computation and `torch` uses all
available threads by default.

You can control the number of threads used by `torch` with:

    torch::torch_set_num_threads(1)
    torch::torch_set_num_interop_threads(1)

## See also

tabnet_fit

## Examples

``` r
library(parsnip)
data("ames", package = "modeldata")
model <- tabnet() %>%
  set_mode("regression") %>%
  set_engine("torch")
model %>%
  fit(Sale_Price ~ ., data = ames)
#> parsnip model object
#> 
#> An `nn_module` containing 10,742 parameters.
#> 
#> ── Modules ─────────────────────────────────────────────────────────────────────
#> • embedder: <embedding_generator> #283 parameters
#> • embedder_na: <na_embedding_generator> #0 parameters
#> • tabnet: <tabnet_no_embedding> #10,458 parameters
#> 
#> ── Parameters ──────────────────────────────────────────────────────────────────
#> • .check: Float [1:1]
```
