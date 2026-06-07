# Apply hierarchy constraints via max-pooling over descendants (MCM)

Given neural network outputs x and ancestor matrix R, enforces that if a
class is predicted positive, all its ancestors must also be positive.
Implements: `final_out[i] = max{x[j] : R[i,j] = 1}`

## Usage

``` r
get_constr_output(x, R)
```

## Arguments

- x:

  A `torch_tensor` of shape `(batch_size, n_classes)`.

- R:

  A `torch_tensor` of shape `(1, n_classes, n_classes)` where
  `R[1, i, j] = 1` iff class `i` is a descendant of class `j`.

## Value

A `torch_tensor` of shape `(batch_size, n_classes)` with constrained
outputs.
