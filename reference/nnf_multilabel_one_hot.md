# Convert class_id tensor to binary one-hot tensor

Transforms a tensor of class indices (one column per hierarchy level)
into a binary tensor where each column corresponds to a class.

## Usage

``` r
nnf_multilabel_one_hot(y, outcomes, device = "cpu")
```

## Arguments

- y:

  A `torch_tensor` of shape `(batch_size, n_levels)` containing 1-based
  class indices.

- outcomes:

  A tibble with factor columns (as from `hardhat::mold()$outcomes`).

- device:

  Torch device.

## Value

A `torch_tensor` of shape `(batch_size, n_classes)` with binary values.
