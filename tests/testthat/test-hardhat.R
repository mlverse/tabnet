test_that("Training regression", {

  data("ames", package = "modeldata")

  x <- ames[-which(names(ames) == "Sale_Price")]
  y <- ames$Sale_Price

  expect_error(
    fit <- tabnet_fit(x, y, epochs = 1),
    regexp = NA
  )

  expect_error(
    fit <- tabnet_fit(Sale_Price ~ ., data = ames, epochs = 1),
    regexp = NA
  )

  expect_error(
    predict(fit, x),
    regexp = NA
  )

  expect_error(
    fit <- tabnet_fit(x, y, epochs = 2, verbose = TRUE),
    regexp = NA
  )
})

test_that("Training classification", {

  data("attrition", package = "modeldata")

  x <- attrition[-which(names(attrition) == "Attrition")]
  y <- attrition$Attrition

  expect_error(
    fit <- tabnet_fit(x, y, epochs = 1),
    regexp = NA
  )

  expect_error(
    predict(fit, x, type = "prob"),
    regexp = NA
  )

  expect_error(
    predict(fit, x),
    regexp = NA
  )

})

test_that("errors when using an argument that do not exist", {

  data("ames", package = "modeldata")

  x <- ames[-which(names(ames) == "Sale_Price")]
  y <- ames$Sale_Price

  expect_error(
    fit <- tabnet_fit(x, y, epochsas = 1),
    "unused argument"
  )

})

test_that("works with validation split", {

  data("attrition", package = "modeldata")

  x <- attrition[-which(names(attrition) == "Attrition")]
  y <- attrition$Attrition

  expect_error(
    fit <- tabnet_fit(x, y, epochs = 1, valid_split = 0.2),
    regexp = NA
  )

  expect_error(
    fit <- tabnet_fit(x, y, epochs = 1, valid_split = 0.2, verbose = TRUE),
    regexp = NA
  )

})

test_that("can train from a recipe", {

  library(recipes)
  data("attrition", package = "modeldata")

  rec <- recipe(Attrition ~ ., data = attrition) %>%
    step_normalize(all_numeric(), -all_outcomes())

  expect_error(
    fit <- tabnet_fit(rec, attrition, epochs = 1, valid_split = 0.25,
                    verbose = TRUE),
    regexp = NA
  )

  expect_error(
    predict(fit, attrition),
    regexp = NA
  )

})

test_that("serialization with saveRDS just works", {

  data("ames", package = "modeldata")

  x <- ames[-which(names(ames) == "Sale_Price")]
  y <- ames$Sale_Price

  fit <- tabnet_fit(x, y, epochs = 1)

  tmp <- tempfile("model", fileext = "rds")
  saveRDS(fit, tmp)

  fit2 <- readRDS(tmp)

  expect_equal(
    predict(fit, ames),
    predict(fit2, ames)
  )

  expect_equal(as.numeric(fit2$fit$network$.check), 1)

})

