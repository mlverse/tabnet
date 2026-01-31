# Parameters for the tabnet model

Parameters for the tabnet model

## Usage

``` r
attention_width(range = c(8L, 64L), trans = NULL)

decision_width(range = c(8L, 64L), trans = NULL)

feature_reusage(range = c(1, 2), trans = NULL)

momentum(range = c(0.01, 0.4), trans = NULL)

mask_type(values = c("sparsemax", "entmax"))

num_independent(range = c(1L, 5L), trans = NULL)

num_shared(range = c(1L, 5L), trans = NULL)

num_steps(range = c(3L, 10L), trans = NULL)
```

## Arguments

- range:

  the default range for the parameter value

- trans:

  whether to apply a transformation to the parameter

- values:

  possible values for factor parameters

  These functions are used with `tune` grid functions to generate
  candidates.

## Value

A `dials` parameter to be used when tuning TabNet models.

## Examples

``` r
  model <- tabnet(attention_width = tune(), feature_reusage = tune(),
    momentum = tune(), penalty = tune(), rate_step_size = tune()) %>%
    parsnip::set_mode("regression") %>%
    parsnip::set_engine("torch")
```
