# Tabnet model

Pretrain the [TabNet: Attentive Interpretable Tabular
Learning](https://arxiv.org/abs/1908.07442) model on the predictor data
exclusively (unsupervised training).

## Usage

``` r
tabnet_pretrain(x, ...)

# Default S3 method
tabnet_pretrain(x, ...)

# S3 method for class 'data.frame'
tabnet_pretrain(
  x,
  y,
  tabnet_model = NULL,
  config = tabnet_config(),
  ...,
  from_epoch = NULL
)

# S3 method for class 'formula'
tabnet_pretrain(
  formula,
  data,
  tabnet_model = NULL,
  config = tabnet_config(),
  ...,
  from_epoch = NULL
)

# S3 method for class 'recipe'
tabnet_pretrain(
  x,
  data,
  tabnet_model = NULL,
  config = tabnet_config(),
  ...,
  from_epoch = NULL
)

# S3 method for class 'Node'
tabnet_pretrain(
  x,
  tabnet_model = NULL,
  config = tabnet_config(),
  ...,
  from_epoch = NULL
)
```

## Arguments

- x:

  Depending on the context:

  - A **data frame** of predictors.

  - A **matrix** of predictors.

  - A **recipe** specifying a set of preprocessing steps created from
    [`recipes::recipe()`](https://recipes.tidymodels.org/reference/recipe.html).

  - A **Node** where tree leaves will be left out, and attributes will
    be used as predictors.

  The predictor data should be standardized (e.g. centered or scaled).
  The model treats categorical predictors internally thus, you don't
  need to make any treatment. The model treats missing values internally
  thus, you don't need to make any treatment.

- ...:

  Model hyperparameters. Any hyperparameters set here will update those
  set by the config argument. See [`tabnet_config()`](tabnet_config.md)
  for a list of all possible hyperparameters.

- y:

  (optional) When `x` is a **data frame** or **matrix**, `y` is the
  outcome

- tabnet_model:

  A pretrained `tabnet_model` object to continue the fitting on. if
  `NULL` (the default) a brand new model is initialized.

- config:

  A set of hyperparameters created using the `tabnet_config` function.
  If no argument is supplied, this will use the default values in
  [`tabnet_config()`](tabnet_config.md).

- from_epoch:

  When a `tabnet_model` is provided, restore the network weights from a
  specific epoch. Default is last available checkpoint for restored
  model, or last epoch for in-memory model.

- formula:

  A formula specifying the outcome terms on the left-hand side, and the
  predictor terms on the right-hand side.

- data:

  When a **recipe** or **formula** is used, `data` is specified as:

  - A **data frame** containing both the predictors and the outcome.

## Value

A TabNet model object. It can be used for serialization, predictions, or
further fitting.

## outcome

Outcome value are accepted here only for consistent syntax with
`tabnet_fit`, but by design the outcome, if present, is ignored during
pre-training.

## pre-training from a previous model

When providing a parent `tabnet_model` parameter, the model pretraining
resumes from that model weights at the following epoch:

- last pretrained epoch for a model already in torch context

- Last model checkpoint epoch for a model loaded from file

- the epoch related to a checkpoint matching or preceding the
  `from_epoch` value if provided The model pretraining metrics append on
  top of the parent metrics in the returned TabNet model.

## Threading

TabNet uses `torch` as its backend for computation and `torch` uses all
available threads by default.

You can control the number of threads used by `torch` with:

    torch::torch_set_num_threads(1)
    torch::torch_set_num_interop_threads(1)

## Examples

``` r
data("ames", package = "modeldata")
pretrained <- tabnet_pretrain(Sale_Price ~ ., data = ames, epochs = 1)
```
