# Plot tabnet_explain mask importance heatmap

Plot tabnet_explain mask importance heatmap

## Usage

``` r
autoplot.tabnet_explain(
  object,
  type = c("mask_agg", "steps"),
  quantile = 1,
  ...
)
```

## Arguments

- object:

  A `tabnet_explain` object as a result of
  [`tabnet_explain()`](tabnet_explain.md).

- type:

  a character value. Either `"mask_agg"` the default, for a single
  heatmap of aggregated mask importance per predictor along the dataset,
  or `"steps"` for one heatmap at each mask step.

- quantile:

  numerical value between 0 and 1. Provides quantile clipping of the
  mask values

- ...:

  not used.

## Value

A `ggplot` object.

## Details

Plot the `tabnet_explain` object mask importance per variable along the
predicted dataset. `type="mask_agg"` output a single heatmap of mask
aggregated values, `type="steps"` provides a plot faceted along the
`n_steps` mask present in the model. `quantile=.995` may be used for
strong outlier clipping, in order to better highlight low values.
`quantile=1`, the default, do not clip any values.

## Examples

``` r
 if (FALSE) { # \dontrun{
library(ggplot2)
data("attrition", package = "modeldata")

## Single-outcome binary classification of `Attrition` in `attrition` dataset
attrition_fit <- tabnet_fit(Attrition ~. , data=attrition, epoch=11)
attrition_explain <- tabnet_explain(attrition_fit, attrition)
# Plot the model aggregated mask interpretation heatmap
autoplot(attrition_explain)

## Multi-outcome regression on `Sale_Price` and `Pool_Area` in `ames` dataset,
data("ames", package = "modeldata")
x <- ames[,-which(names(ames) %in% c("Sale_Price", "Pool_Area"))]
y <- ames[, c("Sale_Price", "Pool_Area")]
ames_fit <- tabnet_fit(x, y, epochs = 1, verbose=TRUE)
ames_explain <- tabnet_explain(ames_fit, x)
autoplot(ames_explain, quantile = 0.99)
} # }
```
