
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
Learning](https://arxiv.org/abs/1908.07442) [(Sercan O. Arik, Tomas
Pfister)](https://doi.org/10.48550/arXiv.1908.07442). It is initially an
R port using the [torch](https://github.com/mlverse/torch) package of
[dreamquark-ai/tabnet](https://github.com/dreamquark-ai/tabnet)
PyTorch’s Tabnet implementation.  
TabNet is augmented with  
- Missing values in predictors.  
- [Coherent Hierarchical Multi-label Classification
Networks](https://proceedings.neurips.cc//paper/2020/file/6dd4e10e3296fa63738371ec0d5df818-Paper.pdf)
[(Eleonora Giunchiglia et Al.)](giunchiglia2020neurips) for hierarchical
outcomes.  
- [InterpreTabNet: Enhancing Interpretability of Tabular Data
Using](https://arxiv.org/abs/2310.02870) [(Shiyun Wa et
Al.)](https://doi.org/10.48550/arXiv.2310.02870)

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

## Basic Binary Classification Example

Here we show a **binary classification** example of the `attrition`
dataset, using a **recipe** for dataset input specification.

``` r
library(tabnet)
suppressPackageStartupMessages(library(recipes))
library(yardstick)
library(ggplot2)
set.seed(1)

data("attrition", package = "modeldata")
test_idx <- sample.int(nrow(attrition), size = 0.2 * nrow(attrition))

train <- attrition[-test_idx,]
test <- attrition[test_idx,]

rec <- recipe(Attrition ~ ., data = train) %>% 
  step_normalize(all_numeric(), -all_outcomes())

fit <- tabnet_fit(rec, train, epochs = 30, valid_split=0.1, learn_rate = 5e-3)
#> [Epoch 001] Loss: 0.887588, Valid loss: 0.945602
#> [Epoch 002] Loss: 0.797204, Valid loss: 0.885115
#> [Epoch 003] Loss: 0.715031, Valid loss: 0.818196
#> [Epoch 004] Loss: 0.656060, Valid loss: 0.850705
#> [Epoch 005] Loss: 0.609140, Valid loss: 0.786945
#> [Epoch 006] Loss: 0.573324, Valid loss: 0.720014
#> [Epoch 007] Loss: 0.537668, Valid loss: 0.675371
#> [Epoch 008] Loss: 0.506530, Valid loss: 0.646100
#> [Epoch 009] Loss: 0.480609, Valid loss: 0.619250
#> [Epoch 010] Loss: 0.457574, Valid loss: 0.595247
#> [Epoch 011] Loss: 0.438565, Valid loss: 0.591589
#> [Epoch 012] Loss: 0.422576, Valid loss: 0.584090
#> [Epoch 013] Loss: 0.408241, Valid loss: 0.573011
#> [Epoch 014] Loss: 0.395664, Valid loss: 0.569446
#> [Epoch 015] Loss: 0.384885, Valid loss: 0.564139
#> [Epoch 016] Loss: 0.376060, Valid loss: 0.560019
#> [Epoch 017] Loss: 0.366509, Valid loss: 0.563200
#> [Epoch 018] Loss: 0.358134, Valid loss: 0.559758
#> [Epoch 019] Loss: 0.350165, Valid loss: 0.558542
#> [Epoch 020] Loss: 0.342638, Valid loss: 0.562761
#> [Epoch 021] Loss: 0.334345, Valid loss: 0.564604
#> [Epoch 022] Loss: 0.327447, Valid loss: 0.564762
#> [Epoch 023] Loss: 0.321357, Valid loss: 0.567150
#> [Epoch 024] Loss: 0.310901, Valid loss: 0.570480
#> [Epoch 025] Loss: 0.303838, Valid loss: 0.570377
#> [Epoch 026] Loss: 0.295977, Valid loss: 0.561524
#> [Epoch 027] Loss: 0.288999, Valid loss: 0.555991
#> [Epoch 028] Loss: 0.282421, Valid loss: 0.550610
#> [Epoch 029] Loss: 0.274548, Valid loss: 0.544487
#> [Epoch 030] Loss: 0.268768, Valid loss: 0.535972
autoplot(fit)
```

<img src="man/figures/README-model-fit-1.png" width="100%" />

The plots gives you an immediate insight about model over-fitting, and
if any, the available model checkpoints available before the
over-fitting

Keep in mind that **regression** as well as **multi-class
classification** are also available, and that you can specify dataset
through **data.frame** and **formula** as well. You will find them in
the package vignettes.

## Model performance results

As the standard method `predict()` is used, you can rely on your usual
metric functions for model performance results. Here we use {yardstick}
:

``` r
metrics <- metric_set(accuracy, precision, recall)
cbind(test, predict(fit, test)) %>% 
  metrics(Attrition, estimate = .pred_class)
#> # A tibble: 3 × 3
#>   .metric   .estimator .estimate
#>   <chr>     <chr>          <dbl>
#> 1 accuracy  binary         0.833
#> 2 precision binary         0.838
#> 3 recall    binary         0.992
  
cbind(test, predict(fit, test, type = "prob")) %>% 
  roc_auc(Attrition, .pred_No)
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.572
```

## Explain model on test-set with attention map

TabNet has intrinsic explainability feature through the visualization of
attention map, either **aggregated**:

``` r
explain <- tabnet_explain(fit, test)
autoplot(explain)
```

<img src="man/figures/README-model-explain-1.png" width="100%" />

or at **each layer** through the `type = "steps"` option:

``` r
autoplot(explain, type = "steps")
```

<img src="man/figures/README-step-explain-1.png" width="100%" />

## Self-supervised pretraining

For cases when a consistent part of your dataset has no outcome, TabNet
offers a self-supervised training step allowing to model to capture
predictors intrinsic features and predictors interactions, upfront the
supervised task.

``` r
pretrain <- tabnet_pretrain(rec, train, epochs = 50, valid_split=0.1, learn_rate = 1e-2)
#> [Epoch 001] Loss: 4.932210, Valid loss: 4.092751
#> [Epoch 002] Loss: 4.106266, Valid loss: 3.355779
#> [Epoch 003] Loss: 3.564717, Valid loss: 2.957145
#> [Epoch 004] Loss: 3.042337, Valid loss: 2.601007
#> [Epoch 005] Loss: 2.739125, Valid loss: 2.351617
#> [Epoch 006] Loss: 2.526936, Valid loss: 2.125806
#> [Epoch 007] Loss: 2.333645, Valid loss: 1.984583
#> [Epoch 008] Loss: 2.198569, Valid loss: 1.878892
#> [Epoch 009] Loss: 2.041425, Valid loss: 1.809348
#> [Epoch 010] Loss: 1.918194, Valid loss: 1.710608
#> [Epoch 011] Loss: 1.845506, Valid loss: 1.637800
#> [Epoch 012] Loss: 1.760037, Valid loss: 1.597510
#> [Epoch 013] Loss: 1.653455, Valid loss: 1.551801
#> [Epoch 014] Loss: 1.696923, Valid loss: 1.514207
#> [Epoch 015] Loss: 1.600405, Valid loss: 1.485546
#> [Epoch 016] Loss: 1.583056, Valid loss: 1.458767
#> [Epoch 017] Loss: 1.507135, Valid loss: 1.434835
#> [Epoch 018] Loss: 1.476803, Valid loss: 1.412581
#> [Epoch 019] Loss: 1.432387, Valid loss: 1.397751
#> [Epoch 020] Loss: 1.440812, Valid loss: 1.384655
#> [Epoch 021] Loss: 1.418823, Valid loss: 1.372710
#> [Epoch 022] Loss: 1.366211, Valid loss: 1.358185
#> [Epoch 023] Loss: 1.356015, Valid loss: 1.344237
#> [Epoch 024] Loss: 1.305629, Valid loss: 1.330525
#> [Epoch 025] Loss: 1.315370, Valid loss: 1.315821
#> [Epoch 026] Loss: 1.311765, Valid loss: 1.303614
#> [Epoch 027] Loss: 1.291411, Valid loss: 1.292028
#> [Epoch 028] Loss: 1.278566, Valid loss: 1.279822
#> [Epoch 029] Loss: 1.255989, Valid loss: 1.267850
#> [Epoch 030] Loss: 1.260587, Valid loss: 1.257541
#> [Epoch 031] Loss: 1.238879, Valid loss: 1.245630
#> [Epoch 032] Loss: 1.226142, Valid loss: 1.232984
#> [Epoch 033] Loss: 1.213936, Valid loss: 1.221010
#> [Epoch 034] Loss: 1.201698, Valid loss: 1.210083
#> [Epoch 035] Loss: 1.191756, Valid loss: 1.199468
#> [Epoch 036] Loss: 1.202733, Valid loss: 1.190049
#> [Epoch 037] Loss: 1.157731, Valid loss: 1.180808
#> [Epoch 038] Loss: 1.149515, Valid loss: 1.171888
#> [Epoch 039] Loss: 1.163042, Valid loss: 1.162936
#> [Epoch 040] Loss: 1.155832, Valid loss: 1.154495
#> [Epoch 041] Loss: 1.140963, Valid loss: 1.146452
#> [Epoch 042] Loss: 1.121486, Valid loss: 1.139101
#> [Epoch 043] Loss: 1.127143, Valid loss: 1.132156
#> [Epoch 044] Loss: 1.130294, Valid loss: 1.126133
#> [Epoch 045] Loss: 1.125776, Valid loss: 1.119981
#> [Epoch 046] Loss: 1.108063, Valid loss: 1.114049
#> [Epoch 047] Loss: 1.114287, Valid loss: 1.108261
#> [Epoch 048] Loss: 1.106954, Valid loss: 1.102695
#> [Epoch 049] Loss: 1.106918, Valid loss: 1.097771
#> [Epoch 050] Loss: 1.106329, Valid loss: 1.093506
autoplot(pretrain)
```

<img src="man/figures/README-step-pretrain-1.png" width="100%" />

The example here is a toy example as the `train` dataset does actually
contain outcomes. The vignette on [Self-supervised training and
fine-tuning](https://mlverse.github.io/tabnet/articles/selfsupervised_training.html)
will gives you the complete correct workflow step-by-step.

## Missing data in predictors

{tabnet} leverage the masking mechanism to deal with missing data, so
you don’t have to remove the entries in your dataset with some missing
values in the predictors variables.

# Comparison with other implementations

| Group            | Feature                              |      {tabnet}      | dreamquark-ai | fast-tabnet |
|------------------|--------------------------------------|:------------------:|:-------------:|:-----------:|
| Input format     | data-frame                           |         ✅         |      ✅       |     ✅      |
|                  | formula                              |         ✅         |               |             |
|                  | recipe                               |         ✅         |               |             |
|                  | Node                                 |         ✅         |               |             |
|                  | missings in predictor                |         ✅         |               |             |
| Output format    | data-frame                           |         ✅         |      ✅       |     ✅      |
|                  | workflow                             |         ✅         |               |             |
| ML Tasks         | self-supervised learning             |         ✅         |      ✅       |             |
|                  | classification (binary, multi-class) |         ✅         |      ✅       |     ✅      |
|                  | regression                           |         ✅         |      ✅       |     ✅      |
|                  | multi-outcome                        |         ✅         |      ✅       |             |
|                  | hierarchical multi-label classif.    |         ✅         |               |             |
| Model management | from / to file                       |         ✅         |      ✅       |      v      |
|                  | resume from snapshot                 |         ✅         |               |             |
|                  | training diagnostic                  |         ✅         |               |             |
| Interpretability | default                              |         ✅         |      ✅       |     ✅      |
|                  | stabilized                           |         ✅         |               |             |
| Performance      |                                      |        1 x         |    2 - 4 x    |             |
| Code quality     | test coverage                        |        85%         |               |             |
|                  | continuous integration               | 4 OS including GPU |               |             |

Alternative TabNet implementation features
