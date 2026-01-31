# Check that Node object names are compliant

Check that Node object names are compliant

## Usage

``` r
check_compliant_node(node)
```

## Arguments

- node:

  the Node object, or a dataframe ready to be parsed by
  [`data.tree::as.Node()`](https://rdrr.io/pkg/data.tree/man/as.Node.html)

## Value

node if it is compliant, else an Error with the column names to fix

## Examples

``` r
library(dplyr)
#> 
#> Attaching package: ‘dplyr’
#> The following objects are masked from ‘package:stats’:
#> 
#>     filter, lag
#> The following objects are masked from ‘package:base’:
#> 
#>     intersect, setdiff, setequal, union
library(data.tree)
data(starwars)
starwars_tree <- starwars %>%
  mutate(pathString = paste("tree", species, homeworld, `name`, sep = "/"))

# pre as.Node() check
try(check_compliant_node(starwars_tree))
#> Error in check_compliant_node(starwars_tree) : 
#>   The attributes or colnames in the provided hierarchical object use the
#> following reserved names: name and height.  Please change those names as they
#> will lead to unexpected tabnet behavior.

# post as.Node() check
check_compliant_node(as.Node(starwars_tree))
```
