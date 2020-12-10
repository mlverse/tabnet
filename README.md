
<!-- README.md is generated from README.Rmd. Please edit that file -->

# tabnet

Work In Progress\!

<!-- badges: start -->

<!-- badges: end -->

An R implementation of: [TabNet: Attentive Interpretable Tabular
Learning](https://arxiv.org/abs/1908.07442). The code in this repository
is an R port of
[dreamquark-ai/tabnet](https://github.com/dreamquark-ai/tabnet)
PyTorch’s implementation using the
[torch](https://github.com/mlverse/torch) package.

## Installation

You can install the development version from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("mlverse/tabnet")
```

## Example

``` r
library(tabnet)
library(recipes)
#> Loading required package: dplyr
#> 
#> Attaching package: 'dplyr'
#> The following objects are masked from 'package:stats':
#> 
#>     filter, lag
#> The following objects are masked from 'package:base':
#> 
#>     intersect, setdiff, setequal, union
#> 
#> Attaching package: 'recipes'
#> The following object is masked from 'package:stats':
#> 
#>     step
library(yardstick)
#> For binary classification, the first factor level is assumed to be the event.
#> Use the argument `event_level = "second"` to alter this as needed.

data("attrition", package = "modeldata")
test_idx <- sample.int(nrow(attrition), size = 0.2 * nrow(attrition))

train <- attrition[-test_idx,]
test <- attrition[test_idx,]

rec <- recipe(Attrition ~ ., data = train) %>% 
  step_normalize(all_numeric(), -all_outcomes())

fit <- tabnet_fit(rec, train, epochs = 50)

cbind(attrition, predict(fit, test)) %>% 
  accuracy(Attrition, .pred_class)
#> # A tibble: 1 x 3
#>   .metric  .estimator .estimate
#>   <chr>    <chr>          <dbl>
#> 1 accuracy binary         0.795
  
cbind(attrition, predict(fit, test, type = "prob")) %>% 
  roc_auc(Attrition, .pred_No)
#> # A tibble: 1 x 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.491
```
