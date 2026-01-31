# TabNet Model Architecture

This is a `nn_module` representing the TabNet architecture from
[Attentive Interpretable Tabular Deep
Learning](https://arxiv.org/abs/1908.07442).

## Usage

``` r
tabnet_nn(
  input_dim,
  output_dim,
  n_d = 8,
  n_a = 8,
  n_steps = 3,
  gamma = 1.3,
  cat_idxs = c(),
  cat_dims = c(),
  cat_emb_dim = 1,
  n_independent = 2,
  n_shared = 2,
  epsilon = 1e-15,
  virtual_batch_size = 128,
  momentum = 0.02,
  mask_type = "sparsemax",
  mask_topk = NULL
)
```

## Arguments

- input_dim:

  Initial number of features.

- output_dim:

  Dimension of network output. Examples : one for regression, 2 for
  binary classification etc.. Vector of those dimensions in case of
  multi-output.

- n_d:

  Dimension of the prediction layer (usually between 4 and 64).

- n_a:

  Dimension of the attention layer (usually between 4 and 64).

- n_steps:

  Number of successive steps in the network (usually between 3 and 10).

- gamma:

  Scaling factor for attention updates (usually between 1 and 2).

- cat_idxs:

  Index of each categorical column in the dataset.

- cat_dims:

  Number of categories in each categorical column.

- cat_emb_dim:

  Size of the embedding of categorical features if int, all categorical
  features will have same embedding size if list of int, every
  corresponding feature will have specific size.

- n_independent:

  Number of independent GLU layer in each GLU block of the encoder.

- n_shared:

  Number of shared GLU layer in each GLU block of the encoder.

- epsilon:

  Avoid log(0), this should be kept very low.

- virtual_batch_size:

  Batch size for Ghost Batch Normalization.

- momentum:

  Numerical value between 0 and 1 which will be used for momentum in all
  batch norm.

- mask_type:

  Either "sparsemax", "entmax" or "entmax15": the sparse masking
  function to use.

- mask_topk:

  the mask top-k value for k-sparsity selection in the mask for
  `sparsemax` and `entmax15`. defaults to 1/4 of last `input_dim` if
  `NULL`. See [entmax15](entmax15.md) for details.
