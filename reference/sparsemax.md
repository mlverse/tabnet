# Sparsemax

Normalizing sparse transform (a la softmax).

## Usage

``` r
sparsemax(dim = -1L)

sparsemax15(dim = -1L, k = NULL)
```

## Arguments

- dim:

  The dimension along which to apply sparsemax.

- k:

  The number of largest elements to partial-sort input over. For optimal
  performance, `k` should be slightly bigger than the expected number of
  non-zeros in the solution. If the solution is more than k-sparse, this
  function is recursively called with a 2\*k schedule. If `NULL`, full
  sorting is performed from the beginning.

## Value

The projection result, such that \\\sum\_{dim} P = 1 \forall dim\\
elementwise.

## Details

Solves the projection:

\\\min_P \|\|input - P\|\|\_2 \text{ s.t. } P \geq0, \sum(P) ==1\\

## Examples

``` r
if (FALSE) { # \dontrun{
input <- torch::torch_randn(10, 5, requires_grad = TRUE)
# create a top3 alpha=1.5 sparsemax on last input dimension
nn_sparsemax <- sparsemax15(dim=1, k=3)
result <- nn_sparsemax(input)
print(result)
} # }
```
