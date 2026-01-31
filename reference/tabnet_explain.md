# Interpretation metrics from a TabNet model

Interpretation metrics from a TabNet model

## Usage

``` r
tabnet_explain(object, new_data)

# Default S3 method
tabnet_explain(object, new_data)

# S3 method for class 'tabnet_fit'
tabnet_explain(object, new_data)

# S3 method for class 'tabnet_pretrain'
tabnet_explain(object, new_data)

# S3 method for class 'model_fit'
tabnet_explain(object, new_data)
```

## Arguments

- object:

  a TabNet fit object

- new_data:

  a data.frame to obtain interpretation metrics.

## Value

Returns a list with

- `M_explain`: the aggregated feature importance masks as detailed in
  TabNet's paper.

- `masks` a list containing the masks for each step.

## Examples

``` r
set.seed(2021)

n <- 256
x <- data.frame(
  x = rnorm(n),
  y = rnorm(n),
  z = rnorm(n)
)

y <- x$x

fit <- tabnet_fit(x, y, epochs = 10,
                  num_steps = 1,
                  batch_size = 512,
                  attention_width = 1,
                  num_shared = 1,
                  num_independent = 1)


 ex <- tabnet_explain(fit, x)
```
