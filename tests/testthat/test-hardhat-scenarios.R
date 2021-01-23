test_that("we can continue training with a additional fit", {

  data("ames", package = "modeldata")

  x <- ames[-which(names(ames) == "Sale_Price")]
  y <- ames$Sale_Price

  fit_1 <- tabnet_fit(x, y, epochs = 1)
  fit_2 <- tabnet_fit(x, y, fit_1, epochs = 1)

  expect_equal(fit_2$fit$config$epoch, 2)
  expect_lte(mean(fit_2$fit$metrics[[2]]$train$loss), mean(fit_2$fit$metrics[[1]]$train$loss))

})

test_that("we can change the tabnet_options between training epoch", {

  fit_3 <- tabnet_fit(x, y, fit_2, epochs = 1, penalty = 0.003, learn_rate = 0.002)

  expect_equal(fit_3$fit$config$epoch, 3)
  expect_lte(mean(fit_3$fit$metrics[[3]]$train$loss), mean(fit_3$fit$metrics[[2]]$train$loss))

})
