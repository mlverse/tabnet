# Turn a Node object into predictor and outcome.

Turn a Node object into predictor and outcome.

## Usage

``` r
node_to_df(x, drop_last_level = TRUE)
```

## Arguments

- x:

  Node object

- drop_last_level:

  TRUE unused

## Value

a named list of x and y, being respectively the predictor data-frame and
the outcomes data-frame, as expected inputs for
[`hardhat::mold()`](https://hardhat.tidymodels.org/reference/mold.html)
function.

## Examples

``` r
library(dplyr)
library(data.tree)
data(starwars)
starwars_tree <- starwars %>%
  mutate(pathString = paste("tree", species, homeworld, `name`, sep = "/")) %>%
  as.Node()
node_to_df(starwars_tree)$x %>% head()
#>   birth_year eye_color
#> 1       19.0      blue
#> 2       41.9    yellow
#> 3       52.0      blue
#> 4       47.0      blue
#> 5       24.0     brown
#> 6       41.9      blue
#>                                                                                             films
#> 1 A New Hope, The Empire Strikes Back, Return of the Jedi, Revenge of the Sith, The Force Awakens
#> 2                    A New Hope, The Empire Strikes Back, Return of the Jedi, Revenge of the Sith
#> 3                                           A New Hope, Attack of the Clones, Revenge of the Sith
#> 4                                           A New Hope, Attack of the Clones, Revenge of the Sith
#> 5                                                                                      A New Hope
#> 6                                   The Phantom Menace, Attack of the Clones, Revenge of the Sith
#>      gender  hair_color homeworld mass    sex skin_color species
#> 1 masculine       blond  Tatooine   77   male       fair   Human
#> 2 masculine        none  Tatooine  136   male      white   Human
#> 3 masculine brown, grey  Tatooine  120   male      light   Human
#> 4  feminine       brown  Tatooine   75 female      light   Human
#> 5 masculine       black  Tatooine   84   male      light   Human
#> 6 masculine       blond  Tatooine   84   male       fair   Human
#>                                                   starships
#> 1                                  X-wing, Imperial shuttle
#> 2                                           TIE Advanced x1
#> 3                                                      <NA>
#> 4                                                      <NA>
#> 5                                                    X-wing
#> 6 Naboo fighter, Trade Federation cruiser, Jedi Interceptor
#>                               vehicles
#> 1   Snowspeeder, Imperial Speeder Bike
#> 2                                 <NA>
#> 3                                 <NA>
#> 4                                 <NA>
#> 5                                 <NA>
#> 6 Zephyr-G swoop bike, XJ-6 airspeeder
node_to_df(starwars_tree)$y %>% head()
#>   level_2  level_3
#> 1   Human Tatooine
#> 2   Human Tatooine
#> 3   Human Tatooine
#> 4   Human Tatooine
#> 5   Human Tatooine
#> 6   Human Tatooine
```
