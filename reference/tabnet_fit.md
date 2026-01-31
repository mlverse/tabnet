# Tabnet model

Fits the [TabNet: Attentive Interpretable Tabular
Learning](https://arxiv.org/abs/1908.07442) model

## Usage

``` r
tabnet_fit(x, ...)

# Default S3 method
tabnet_fit(x, ...)

# S3 method for class 'data.frame'
tabnet_fit(
  x,
  y,
  tabnet_model = NULL,
  config = tabnet_config(),
  ...,
  from_epoch = NULL,
  weights = NULL
)

# S3 method for class 'formula'
tabnet_fit(
  formula,
  data,
  tabnet_model = NULL,
  config = tabnet_config(),
  ...,
  from_epoch = NULL,
  weights = NULL
)

# S3 method for class 'recipe'
tabnet_fit(
  x,
  data,
  tabnet_model = NULL,
  config = tabnet_config(),
  ...,
  from_epoch = NULL,
  weights = NULL
)

# S3 method for class 'Node'
tabnet_fit(
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

  - A **Node** where tree will be used as hierarchical outcome, and
    attributes will be used as predictors.

  The predictor data should be standardized (e.g. centered or scaled).
  The model treats categorical predictors internally thus, you don't
  need to make any treatment. The model treats missing values internally
  thus, you don't need to make any treatment.

- ...:

  Model hyperparameters. Any hyperparameters set here will update those
  set by the config argument. See [`tabnet_config()`](tabnet_config.md)
  for a list of all possible hyperparameters.

- y:

  When `x` is a **data frame** or **matrix**, `y` is the outcome
  specified as:

  - A **data frame** with 1 or many numeric column (regression) or 1 or
    many categorical columns (classification) .

  - A **matrix** with 1 column.

  - A **vector**, either numeric or categorical.

- tabnet_model:

  A previously fitted `tabnet_model` object to continue the fitting on.
  if `NULL` (the default) a brand new model is initialized.

- config:

  A set of hyperparameters created using the `tabnet_config` function.
  If no argument is supplied, this will use the default values in
  [`tabnet_config()`](tabnet_config.md).

- from_epoch:

  When a `tabnet_model` is provided, restore the network weights from a
  specific epoch. Default is last available checkpoint for restored
  model, or last epoch for in-memory model.

- weights:

  Unused. Placeholder for hardhat::importance_weight() variables.

- formula:

  A formula specifying the outcome terms on the left-hand side, and the
  predictor terms on the right-hand side.

- data:

  When a **recipe** or **formula** is used, `data` is specified as:

  - A **data frame** containing both the predictors and the outcome.

## Value

A TabNet model object. It can be used for serialization, predictions, or
further fitting.

## Fitting a pre-trained model

When providing a parent `tabnet_model` parameter, the model fitting
resumes from that model weights at the following epoch:

- last fitted epoch for a model already in torch context

- Last model checkpoint epoch for a model loaded from file

- the epoch related to a checkpoint matching or preceding the
  `from_epoch` value if provided The model fitting metrics append on top
  of the parent metrics in the returned TabNet model.

## Multi-outcome

TabNet allows multi-outcome prediction, which is usually named
[multi-label
classification](https://en.wikipedia.org/wiki/Multi-label_classification)
or multi-output regression when outcomes are numerical. Multi-outcome
currently expect outcomes to be either all numeric or all categorical.

## Threading

TabNet uses `torch` as its backend for computation and `torch` uses all
available threads by default.

You can control the number of threads used by `torch` with:

    torch::torch_set_num_threads(1)
    torch::torch_set_num_interop_threads(1)

## Examples

``` r
if (FALSE) { # \dontrun{
data("ames", package = "modeldata")
data("attrition", package = "modeldata")

## Single-outcome regression using formula specification
fit <- tabnet_fit(Sale_Price ~ ., data = ames, epochs = 4)

## Single-outcome classification using data-frame specification
attrition_x <- attrition[ids,-which(names(attrition) == "Attrition")]
fit <- tabnet_fit(attrition_x, attrition$Attrition, epochs = 4, verbose = TRUE)

## Multi-outcome regression on `Sale_Price` and `Pool_Area` in `ames` dataset using formula,
ames_fit <- tabnet_fit(Sale_Price + Pool_Area ~ ., data = ames, epochs = 4, valid_split = 0.2)

## Multi-label classification on `Attrition` and `JobSatisfaction` in
## `attrition` dataset using recipe
library(recipes)
rec <- recipe(Attrition + JobSatisfaction ~ ., data = attrition) %>%
  step_normalize(all_numeric(), -all_outcomes())

attrition_fit <- tabnet_fit(rec, data = attrition, epochs = 4, valid_split = 0.2)

## Hierarchical classification on  `acme`
data(acme, package = "data.tree")

acme_fit <- tabnet_fit(acme, epochs = 4, verbose = TRUE)

# Note: Model's number of epochs should be increased for publication-level results.
} # }
```
