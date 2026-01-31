# Determine the minimum set of model fits

`min_grid()` determines exactly what models should be fit in order to
evaluate the entire set of tuning parameter combinations. This is for
internal use only and the API may change in the near future.

## Usage

``` r
# S3 method for class 'tabnet'
min_grid(x, grid, ...)
```

## Arguments

- x:

  A model specification.

- grid:

  A tibble with tuning parameter combinations.

- ...:

  Not currently used.

## Value

A tibble with the minimum tuning parameters to fit and an additional
list column with the parameter combinations used for prediction.

## Details

[`fit_max_value()`](https://tune.tidymodels.org/reference/min_grid.html)
can be used in other packages to implement a
[`min_grid()`](https://generics.r-lib.org/reference/min_grid.html)
method.

## Examples

``` r
library(dials)
#> Loading required package: scales
#> 
#> Attaching package: ‘dials’
#> The following objects are masked from ‘package:tabnet’:
#> 
#>     momentum, penalty
library(tune)
library(parsnip)

tabnet_spec <- tabnet(decision_width = tune(), attention_width = tune()) %>%
  set_mode("regression") %>%
  set_engine("torch")

tabnet_grid <-
  tabnet_spec %>%
  extract_parameter_set_dials() %>%
  grid_regular(levels = 3)

min_grid(tabnet_spec, tabnet_grid)
#> # A tibble: 9 × 3
#>   decision_width attention_width .submodels
#>            <int>           <int> <list>    
#> 1              8               8 <list [0]>
#> 2             36               8 <list [0]>
#> 3             64               8 <list [0]>
#> 4              8              36 <list [0]>
#> 5             36              36 <list [0]>
#> 6             64              36 <list [0]>
#> 7              8              64 <list [0]>
#> 8             36              64 <list [0]>
#> 9             64              64 <list [0]>
```
