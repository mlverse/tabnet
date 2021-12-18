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

  expect_error(
    miss_pretrain <- tabnet_pretrain(x_missing, y, epochs = 1),
    regexp = NA
  )

  # categorical missing
  x_missing <- x
  x_missing[1,"BusinessTravel"] <- NA

  expect_error(
    miss_pretrain <- tabnet_pretrain(x_missing, y, epochs = 1),
    regexp = NA
  )

  # no error when missing in outcome
  expect_error(
    miss_pretrain <- tabnet_pretrain(x, y_missing, epochs = 1),
    regexp = NA
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

  expect_error(
    miss_fit <- tabnet_fit(x_missing, y, epochs = 1),
    regexp = NA
  )

  # categorical missing
  x_missing <- x
  x_missing[1,"BusinessTravel"] <- NA

  expect_error(
    miss_fit <- tabnet_fit(x_missing, y, epochs = 1),
    regexp = NA
  )

  # missing in outcome
  expect_error(
    miss_fit <- tabnet_fit(x, y_missing, epochs = 1),
    regexp = "missing"
  )

})

test_that("predict data-frame with missing value fails with explicit message", {

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
  expect_error(
    predict(fit, x_missing),
    regexp = "missing"
  )
  # categorical missing
  x_missing <- x
  x_missing[1,"BusinessTravel"] <- NA

  # predict
  expect_error(
    predict(fit, x_missing),
    regexp = "missing"
  )

})
