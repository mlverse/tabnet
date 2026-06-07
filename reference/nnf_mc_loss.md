# Max-Constraint Margin Loss (functional)

Computes the hierarchy-constrained loss for multi-label classification.
Enforces that if a class is predicted positive, all its ancestors must
also be positive, using the ancestor matrix R.

## Usage

``` r
nnf_mc_loss(
  output,
  target,
  R,
  to_eval = NULL,
  criterion = nnf_binary_cross_entropy_with_logits
)
```

## Arguments

- output:

  A `torch_tensor` of raw network outputs (pre-sigmoid), shape
  `(batch_size, n_classes)`.

- target:

  Binary target labels, shape `(batch_size, n_classes)`.

- R:

  Ancestor matrix tensor of shape `(1, n_classes, n_classes)` where
  `R[1, i, j] = 1` iff class `i` is a descendant of class `j`.

- to_eval:

  Optional logical tensor of shape `(n_classes,)` indicating which
  classes to include in the loss computation. If `NULL`, all classes are
  evaluated.

- criterion:

  Loss function to apply after constraint propagation. Default:
  `nnf_binary_cross_entropy_with_logits` (expects raw logits).

## Value

A scalar `torch_tensor` containing the computed loss, or a tensor of
shape `(batch_size, n_classes)` if `reduction = "none"`.

## Details

The loss combines constrained outputs differently for positive and
negative labels:

- For positive labels: uses constrained output of label-weighted
  predictions

- For negative labels: uses constrained raw predictions (penalizes
  ancestor violations)

## See also

[`nn_mc_loss()`](nn_mc_loss.md),
[`get_constr_output()`](get_constr_output.md)
