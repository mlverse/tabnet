# Predict using `tabnet`

Predict using `tabnet`

## Usage

``` r
# S3 method for class 'tabnet_fit'
predict(object, new_data, type = NULL, ..., epoch = NULL)

# S3 method for class 'tabnet_fit'
augment(x, new_data, ...)
```

## Arguments

- object, x:

  A `tabnet_fit` object.

- new_data:

  A data frame or matrix of new predictors.

- type:

  expected outcome type within `c("numeric", "prob", "class")`.

- ...:

  Not used, but required for extensibility.

- epoch:

  the epoch of an existing checkpoint to infer from.

## Value

[`predict()`](https://rdrr.io/r/stats/predict.html) returns a tibble of
predictions and
[`augment()`](https://generics.r-lib.org/reference/augment.html) appends
the columns in `new_data`. In either case, the number of rows in the
tibble is guaranteed to be the same as the number of rows in `new_data`.

For regression data, the prediction is in the column `.pred`. For
classification, the class predictions are in `.pred_class` and the
probability estimates are in columns with the pattern `.pred_{level}`
where `level` is the levels of the outcome factor vector.

## Examples

``` r
# Minimal example for quick execution
car_split <- rsample::initial_split(mtcars[ 1:6,   ])

if (FALSE) { # \dontrun{
# Fit
if (torch_is_installed() & interactive()) {
 mod <- tabnet_fit(mpg ~ cyl + log(drat), training(car_split))

 # Predict
 predict(mod, testing(car_split))
 augment(mod, testing(car_split))
}
} # }
```
