# Optimal threshold (tau) computation for 1.5-entmax

Optimal threshold (tau) computation for 1.5-entmax

## Usage

``` r
get_tau(input, dim = -1L, k = NULL)
```

## Arguments

- input:

  The input tensor to compute thresholds over.

- dim:

  The dimension along which to apply 1.5-entmax. Default is -1.

- k:

  The number of largest elements to partial-sort over. For optimal
  performance, should be slightly bigger than the expected number of
  non-zeros in the solution. If the solution is more than k-sparse, this
  function is recursively called with a 2\*k schedule. If `NULL`, full
  sorting is performed from the beginning. Default is NULL.

## Value

The threshold value for each vector, with all but the `dim` dimension
intact.
