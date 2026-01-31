# Configuration for TabNet models

Configuration for TabNet models

## Usage

``` r
tabnet_config(
  batch_size = 1024^2,
  penalty = 0.001,
  clip_value = NULL,
  loss = "auto",
  epochs = 5,
  drop_last = FALSE,
  decision_width = NULL,
  attention_width = NULL,
  num_steps = 3,
  feature_reusage = 1.3,
  mask_type = "sparsemax",
  mask_topk = NULL,
  virtual_batch_size = 256^2,
  valid_split = 0,
  learn_rate = 0.02,
  optimizer = "adam",
  lr_scheduler = NULL,
  lr_decay = 0.1,
  step_size = 30,
  checkpoint_epochs = 10,
  cat_emb_dim = 1,
  num_independent = 2,
  num_shared = 2,
  num_independent_decoder = 1,
  num_shared_decoder = 1,
  momentum = 0.02,
  pretraining_ratio = 0.5,
  verbose = FALSE,
  device = "auto",
  importance_sample_size = NULL,
  early_stopping_monitor = "auto",
  early_stopping_tolerance = 0,
  early_stopping_patience = 0L,
  num_workers = 0L,
  skip_importance = FALSE
)
```

## Arguments

- batch_size:

  (int) Number of examples per batch, large batch sizes are recommended.
  (default: 1024^2)

- penalty:

  This is the extra sparsity loss coefficient as proposed in the
  original paper. The bigger this coefficient is, the sparser your model
  will be in terms of feature selection. Depending on the difficulty of
  your problem, reducing this value could help (default 1e-3).

- clip_value:

  If a num is given this will clip the gradient at clip_value. Pass
  `NULL` to not clip.

- loss:

  (character or function) Loss function for training (default to mse for
  regression and cross entropy for classification)

- epochs:

  (int) Number of training epochs.

- drop_last:

  (logical) Whether to drop last batch if not complete during training

- decision_width:

  (int) Width of the decision prediction layer. Bigger values gives more
  capacity to the model with the risk of overfitting. Values typically
  range from 8 to 64.

- attention_width:

  (int) Width of the attention embedding for each mask. According to the
  paper n_d = n_a is usually a good choice. (default=8)

- num_steps:

  (int) Number of steps in the architecture (usually between 3 and 10)

- feature_reusage:

  (num) This is the coefficient for feature reusage in the masks. A
  value close to 1 will make mask selection least correlated between
  layers. Values range from 1 to 2.

- mask_type:

  (character) Final layer of feature selector in the
  attentive_transformer block, either `"sparsemax"`, `"entmax"` or
  `"entmax15"`.Defaults to `"sparsemax"`.

- mask_topk:

  (int) mask sparsity top-k for `sparsemax15` and `entmax15.` See
  [`entmax15()`](entmax15.md) for detail.

- virtual_batch_size:

  (int) Size of the mini batches used for "Ghost Batch Normalization"
  (default=256^2)

- valid_split:

  In \[0, 1). The fraction of the dataset used for validation. (default
  = 0 means no split)

- learn_rate:

  initial learning rate for the optimizer.

- optimizer:

  the optimization method. currently only `"adam"` is supported, you can
  also pass any torch optimizer function.

- lr_scheduler:

  if `NULL`, no learning rate decay is used. If "step" decays the
  learning rate by `lr_decay` every `step_size` epochs. If
  "reduce_on_plateau" decays the learning rate by `lr_decay` when no
  improvement after `step_size` epochs. It can also be a
  [torch::lr_scheduler](https://torch.mlverse.org/docs/reference/lr_scheduler.html)
  function that only takes the optimizer as parameter. The `step` method
  is called once per epoch.

- lr_decay:

  multiplies the initial learning rate by `lr_decay` every `step_size`
  epochs. Unused if `lr_scheduler` is a
  [`torch::lr_scheduler`](https://torch.mlverse.org/docs/reference/lr_scheduler.html)
  or `NULL`.

- step_size:

  the learning rate scheduler step size. Unused if `lr_scheduler` is a
  [`torch::lr_scheduler`](https://torch.mlverse.org/docs/reference/lr_scheduler.html)
  or `NULL`.

- checkpoint_epochs:

  checkpoint model weights and architecture every `checkpoint_epochs`.
  (default is 10). This may cause large memory usage. Use `0` to disable
  checkpoints.

- cat_emb_dim:

  Size of the embedding of categorical features. If int, all categorical
  features will have same embedding size, if list of int, every
  corresponding feature will have specific embedding size.

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

- momentum:

  Momentum for batch normalization, typically ranges from 0.01 to 0.4
  (default=0.02)

- pretraining_ratio:

  Ratio of features to mask for reconstruction during pretraining.
  Ranges from 0 to 1 (default=0.5)

- verbose:

  (logical) Whether to print progress and loss values during training.

- device:

  the device to use for training. "cpu" or "cuda". The default ("auto")
  uses to "cuda" if it's available, otherwise uses "cpu".

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

- num_workers:

  (int, optional): how many subprocesses to use for data loading. 0
  means that the data will be loaded in the main process. (default: `0`)

- skip_importance:

  if feature importance calculation should be skipped (default: `FALSE`)

## Value

A named list with all hyperparameters of the TabNet implementation.

## Examples

``` r
data("ames", package = "modeldata")

# change the model config for an faster ignite optimizer
config <- tabnet_config(optimizer = torch::optim_ignite_adamw)

## Single-outcome regression using formula specification
fit <- tabnet_fit(Sale_Price ~ ., data = ames, epochs = 1, config = config)
```
