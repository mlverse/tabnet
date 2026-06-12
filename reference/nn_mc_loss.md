# Max-Constraint Margin Loss (module)

Module wrapper for [`nnf_mc_loss()`](nnf_mc_loss.md) with configurable
parameters. Stores the ancestor matrix R and evaluation mask for reuse
across batches.

## Usage

``` r
nn_mc_loss(
  R,
  to_eval = NULL,
  criterion = nnf_binary_cross_entropy_with_logits,
  reduction = "mean"
)
```

## Arguments

- R:

  Ancestor matrix tensor of shape `(1, n_classes, n_classes)`.

- to_eval:

  Optional logical tensor of shape `(n_classes,)` indicating which
  classes to include in loss computation.

- criterion:

  Loss function module or functional to apply after constraint
  propagation. Default: `nn_binary_cross_entropy_with_logits()`.

- reduction:

  (string, optional): Reduction method: `'none'` \| `'mean'` \| `'sum'`.

## Shape

- Input `output`: \\(N, C)\\ where N = batch size, C = number of classes

- Input `target`: \\(N, C)\\, same shape as output, binary values

- Output: scalar by default. If `reduction = "none"`, then \\(N, C')\\
  where C' is the number of evaluated classes

## See also

[`nnf_mc_loss()`](nnf_mc_loss.md),
[`build_ancestor_matrix_from_outcomes()`](build_ancestor_matrix_from_outcomes.md),
[`get_constr_output()`](get_constr_output.md)

## Examples

``` r
if (FALSE) { # \dontrun{
# Build ancestor matrix from hierarchy
R <- build_ancestor_matrix_from_outcomes(my_tree, processed$outcomes, device = "cuda")

# Create loss module
loss_fn <- nn_mc_loss(R = R, reduction = "mean")

# Forward pass
output <- model(x)  # (batch, n_classes)
loss <- loss_fn(output, labels)
loss$backward()
} # }
```
