test_that("Training regression", {

  data("ames", package = "modeldata")

  x <- ames[-which(names(ames) == "Sale_Price")]
  y <- ames$Sale_Price

  config <- tabnet_config(epochs = 1)

  expect_error(
    fit <- tabnet_fit(x, y, config = config),
    regexp = NA
  )

  expect_error(
    fit <- tabnet_fit(Sale_Price ~ ., data = ames, config = config),
    regexp = NA
  )

  expect_error(
    predict(fit, x),
    regexp = NA
  )
})

test_that("Training classification", {

  data("attrition", package = "modeldata")

  x <- attrition[-which(names(attrition) == "Attrition")]
  y <- attrition$Attrition

  config <- tabnet_config(epochs = 1)

  expect_error(
    fit <- tabnet_fit(x, y, config = config),
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


