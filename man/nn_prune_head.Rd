% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/hardhat.R
\name{nn_prune_head.tabnet_fit}
\alias{nn_prune_head.tabnet_fit}
\alias{nn_prune_head.tabnet_pretrain}
\title{Prune top layer(s) of a tabnet network}
\usage{
\method{nn_prune_head}{tabnet_fit}(x, head_size)

\method{nn_prune_head}{tabnet_pretrain}(x, head_size)
}
\arguments{
\item{x}{nn_network to prune}

\item{head_size}{number of nn_layers to prune, should be less than 2}
}
\value{
a tabnet network with the top nn_layer removed
}
\description{
Prune \code{head_size} last layers of a tabnet network in order to
use the pruned module as a sequential embedding module.
}
\examples{
\dontshow{if ((torch::torch_is_installed())) (if (getRversion() >= "3.4") withAutoprint else force)(\{ # examplesIf}
data("ames", package = "modeldata")
x <- ames[,-which(names(ames) == "Sale_Price")]
y <- ames$Sale_Price
# pretrain a tabnet model on ames dataset
ames_pretrain <- tabnet_pretrain(x, y, epoch = 2, checkpoint_epochs = 1)
# prune classification head to get an embedding model
pruned_pretrain <- torch::nn_prune_head(ames_pretrain, 1)
\dontshow{\}) # examplesIf}
}
