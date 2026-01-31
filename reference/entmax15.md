# Alpha-entmax

With alpha = 1.5 and normalizing sparse transform (a la softmax).

## Usage

``` r
entmax(dim = -1)

entmax15(dim = -1L, k = NULL)
```

## Arguments

- dim:

  The dimension along which to apply 1.5-entmax.

- k:

  The number of largest elements to partial-sort input over. For optimal
  performance, should be slightly bigger than the expected number of
  non-zeros in the solution. If the solution is more than k-sparse, this
  function is recursively called with a 2\*k schedule. If `NULL`, full
  sorting is performed from the beginning.

## Value

The projection result P of the same shape as input, such that
\\\sum\_{dim} P = 1 \forall dim\\ elementwise.

## Details

Solves the optimization problem: \\\max_p \<input, P\> - H\_{1.5}(P)
\text{ s.t. } P \geq 0, \sum(P) == 1\\ where \\H\_{1.5}(P)\\ is the
Tsallis alpha-entropy with \\\alpha=1.5\\.

## Examples

``` r
if (FALSE) { # \dontrun{
input <- torch::torch_randn(10,5, requires_grad = TRUE)
# create a top3 alpha=1.5 entmax on last input dimension
nn_entmax <- entmax15(dim=-1L, k = 3)
result <- nn_entmax(input)
} # }
```
