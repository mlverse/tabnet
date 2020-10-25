test_that("Training regression", {

  data("ames", package = "modeldata")

  x <- ames[-which(names(ames) == "Sale_Price")]
  y <- ames$Sale_Price

  fit <- tabnet_fit(x, y, config = tabnet_config())


})
