# Prune top layer(s) of a tabnet network

Prune `head_size` last layers of a tabnet network in order to use the
pruned module as a sequential embedding module.

## Usage

``` r
# S3 method for class 'tabnet_fit'
nn_prune_head(x, head_size)

# S3 method for class 'tabnet_pretrain'
nn_prune_head(x, head_size)
```

## Arguments

- x:

  nn_network to prune

- head_size:

  number of nn_layers to prune, should be less than 2

## Value

a tabnet network with the top nn_layer removed

## Examples

``` r
data("ames", package = "modeldata")
x <- ames[,-which(names(ames) == "Sale_Price")]
y <- ames$Sale_Price
# pretrain a tabnet model on ames dataset
ames_pretrain <- tabnet_pretrain(x, y, epoch = 2, checkpoint_epochs = 1)
# prune classification head to get an embedding model
pruned_pretrain <- torch::nn_prune_head(ames_pretrain, 1)
```
