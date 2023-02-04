
<!-- README.md is generated from README.Rmd. Please edit that file -->

# tabnet

<!-- badges: start -->

[![R build
status](https://github.com/mlverse/tabnet/workflows/R-CMD-check/badge.svg)](https://github.com/mlverse/tabnet/actions)
[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html)
[![CRAN
status](https://www.r-pkg.org/badges/version/tabnet)](https://CRAN.R-project.org/package=tabnet)
[![](https://cranlogs.r-pkg.org/badges/tabnet)](https://cran.r-project.org/package=tabnet)
[![Discord](https://img.shields.io/discord/837019024499277855?logo=discord)](https://discord.com/invite/s3D5cKhBkx)
<!-- badges: end -->

An R implementation of: [TabNet: Attentive Interpretable Tabular
Learning](https://arxiv.org/abs/1908.07442). The code in this repository
is an R port of
[dreamquark-ai/tabnet](https://github.com/dreamquark-ai/tabnet)
PyTorch’s implementation using the
[torch](https://github.com/mlverse/torch) package.

## Installation

You can install the released version from CRAN with:

``` r
install.packages("tabnet")
```

The development version can be installed from
[GitHub](https://github.com/mlverse/tabnet) with:

``` r
# install.packages("remotes")
remotes::install_github("mlverse/tabnet")
```

## Basic Example

``` r
library(tabnet)
suppressPackageStartupMessages(library(recipes))
library(yardstick)
#> For binary classification, the first factor level is assumed to be the event.
#> Use the argument `event_level = "second"` to alter this as needed.
library(ggplot2)
set.seed(1)

data("attrition", package = "modeldata")
test_idx <- sample.int(nrow(attrition), size = 0.2 * nrow(attrition))

train <- attrition[-test_idx,]
test <- attrition[test_idx,]

rec <- recipe(Attrition ~ ., data = train) %>% 
  step_normalize(all_numeric(), -all_outcomes())

fit <- tabnet_fit(rec, train, epochs = 30, valid_split=0.1)
autoplot(fit)
#> New names:
#> • `` -> `...1`
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> New names:
#> • `` -> `...1`
```

<img src="man/figures/README-model fit-1.png" width="100%" />

## Results

``` r
metrics <- metric_set(accuracy, precision, recall)
cbind(test, predict(fit, test)) %>% 
  metrics(Attrition, estimate = .pred_class)
#> # A tibble: 3 × 3
#>   .metric   .estimator .estimate
#>   <chr>     <chr>          <dbl>
#> 1 accuracy  binary         0.813
#> 2 precision binary         0.840
#> 3 recall    binary         0.959
  
cbind(test, predict(fit, test, type = "prob")) %>% 
  roc_auc(Attrition, .pred_No)
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.587
```

## Explain model on test-set with attention map

either aggregated

``` r
explain <- tabnet_explain(fit, test)
autoplot(explain)
```

<img src="man/figures/README-model explain-1.png" width="100%" />

or at each layer

``` r
autoplot(explain, type="steps")
```

<img src="man/figures/README-step explain-1.png" width="100%" />
