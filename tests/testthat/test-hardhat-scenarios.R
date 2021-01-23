

test_that("we can continue training with a additional fit", {

  data("ames", package = "modeldata")

  x <- ames[-which(names(ames) == "Sale_Price")]
  y <- ames$Sale_Price

  fit_1 <- tabnet_fit(x, y, epochs = 1)
  fit_2 <- tabnet_fit(x, y, fit_1, epochs = 1)

  expect_equal(fit_2$fit$config$epoch, 2)
  expect_lte(mean(fit_2$fit$metrics[[2]]$train$loss), mean(fit_2$fit$metrics[[1]]$train$loss))

})

