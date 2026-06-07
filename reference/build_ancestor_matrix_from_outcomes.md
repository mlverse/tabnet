# Build ancestor matrix aligned with observed outcome classes

Extracts class names from the outcome tibble (factor levels) and builds
the ancestor matrix only for classes that actually appear in the data.

## Usage

``` r
build_ancestor_matrix_from_outcomes(x, outcomes, device = "cpu")
```

## Arguments

- x:

  A [`data.tree::Node`](https://rdrr.io/pkg/data.tree/man/Node.html)
  object.

- outcomes:

  A tibble with factor columns (one per hierarchy level), as returned by
  `hardhat::mold()$outcomes`.

- device:

  Torch device ("cpu" or "cuda").

## Value

A `torch_tensor` of shape `(1, n_classes, n_classes)`.
