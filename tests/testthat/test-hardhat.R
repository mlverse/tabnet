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

  expect_error(
    fit <- tabnet_fit(x, y, epochsas = 1),
    "Argument"
  )

})

test_that("works with validation split", {

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


