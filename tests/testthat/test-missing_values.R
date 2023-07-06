test_that("pretrain accepts missing value in predictors and (unused) outcome", {

  data("attrition", package = "modeldata")
  ids <- sample(nrow(attrition), 256)

  x <- attrition[ids,-which(names(attrition) == "Attrition")]
  y <- attrition[ids,]$Attrition
  y_missing <- y
  y_missing[1] <- NA

  # numerical missing
  x_missing <- x
  x_missing[1,"Age"] <- NA

  expect_no_error(
    miss_pretrain <- tabnet_pretrain(x_missing, y, epochs = 1)
  )

  # categorical missing
  x_missing <- x
  x_missing[1,"BusinessTravel"] <- NA

  expect_no_error(
    miss_pretrain <- tabnet_pretrain(x_missing, y, epochs = 1)
  )

  # no error when missing in outcome
  expect_no_error(
    miss_pretrain <- tabnet_pretrain(x, y_missing, epochs = 1)
  )

})


test_that("fit accept missing value in predictor, not in outcome", {

  data("attrition", package = "modeldata")
  ids <- sample(nrow(attrition), 256)

  x <- attrition[ids,-which(names(attrition) == "Attrition")]
  y <- attrition[ids,]$Attrition
  y_missing <- y
  y_missing[1] <- NA

  # numerical missing
  x_missing <- x
  x_missing[1,"Age"] <- NA

  expect_no_error(
    miss_fit <- tabnet_fit(x_missing, y, epochs = 1)
  )

  # categorical missing
  x_missing <- x
  x_missing[1,"BusinessTravel"] <- NA

  expect_no_error(
    miss_fit <- tabnet_fit(x_missing, y, epochs = 1)
  )

  # missing in outcome
  expect_error(
    miss_fit <- tabnet_fit(x, y_missing, epochs = 1),
    regexp = "missing"
  )

})

test_that("fit accept missing value in `Node` predictor", {
  # fix to https://github.com/mlverse/tabnet/issues/125
  library(data.tree)
  data(starwars, package = "dplyr")

  starwars_tree <- starwars %>%
    rename(`_name` = "name", `_height` = "height") %>%
    mutate(pathString = paste("StarWars_characters", species, homeworld, `_name`, sep = "/")) %>%
    as.Node()

  expect_no_error(
    miss_fit <- tabnet_fit(starwars_tree, epochs = 1, cat_emb_dim = 2)
  )

  expect_no_error(
    miss_pred <- predict(miss_fit, starwars_tree)
  )

})

test_that("predict data-frame accept missing value in predictor", {

  data("attrition", package = "modeldata")
  ids <- sample(nrow(attrition), 256)

  x <- attrition[ids,-which(names(attrition) == "Attrition")]
  y <- attrition[ids,]$Attrition
  #
  fit <- tabnet_fit(x, y, epochs = 1)

  # numerical missing
  x_missing <- x
  x_missing[1,"Age"] <- NA

  # predict with numerical missing
  expect_no_error(
    predict(fit, x_missing),
  )
  # categorical missing
  x_missing <- x
  x_missing[1,"BusinessTravel"] <- NA

  # predict with categorical missing
  expect_no_error(
    predict(fit, x_missing)
  )

})

test_that("inference works with missings in the response vector", {

  data("attrition", package = "modeldata")
  ids <- sample(nrow(attrition), 256)

  rec <- recipe(EnvironmentSatisfaction ~ ., data = attrition[ids, ]) %>%
    step_normalize(all_numeric(), -all_outcomes())
  fit <- tabnet_fit(rec, attrition, epochs = 1, valid_split = 0.25,
                    verbose = TRUE)
  # predict with empty vector
  attrition[["EnvironmentSatisfaction"]] <-NA
  expect_no_error(
    predict(fit, attrition)
  )

  # predict with wrong class
  attrition[["EnvironmentSatisfaction"]] <-NA_character_
  expect_no_error(
    predict(fit, attrition)
  )

  # predict with list column
  attrition[["EnvironmentSatisfaction"]] <- list(NA)
  expect_no_error(
    predict(fit, attrition)
  )

})

test_that("explain works with missings in predictors", {

  data("attrition", package = "modeldata")
  ids <- sample(nrow(attrition), 256)

  x <- attrition[ids,-which(names(attrition) == "Attrition")]
  y <- attrition[ids,]$Attrition
  #
  fit <- tabnet_fit(x, y, epochs = 1)

  # numerical missing
  x_missing <- x
  x_missing[1,"Age"] <- NA

  # explain with numerical missing
  expect_no_error(
    tabnet_explain(fit, x_missing)
  )
  # categorical missing
  x_missing <- x
  x_missing[1,"BusinessTravel"] <- NA

  # explain with categorical missing
  expect_no_error(
    tabnet_explain(fit, x_missing)
  )
})
