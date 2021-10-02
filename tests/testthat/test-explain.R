test_that("explain", {

  set.seed(2)
  torch::torch_manual_seed(2)

  n <- 1000
  x <- data.frame(
    x = rnorm(n),
    y = rnorm(n),
    z = rnorm(n)
  )

  y <- x$x

  fit <- tabnet_fit(x, y, epochs = 10,
                    num_steps = 1,
                    batch_size = 512,
                    attention_width = 1,
                    num_shared = 1,
                    num_independent = 1)

  expect_equal(which.max(fit$fit$importances$importance), 1)
  expect_equal(fit$fit$importances$variables, colnames(x))

  ex <- tabnet_explain(fit, x)

  expect_length(ex, 2)
  expect_length(ex[[2]], 1)
  expect_equal(nrow(ex[[1]]), nrow(x))
  expect_equal(nrow(ex[[2]][[1]]), nrow(x))

})

test_that("support for vip", {

  skip_if_not_installed("vip")

  n <- 1000
  x <- data.frame(
    x = runif(n),
    y = runif(n),
    z = runif(n)
  )

  y <- x$x

  fit <- tabnet_fit(x, y, epochs = 1,
                    num_steps = 1,
                    batch_size = 512,
                    attention_width = 1,
                    num_shared = 1,
                    num_independent = 1)

  expect_error(vip::vip(fit), regexp = NA)

})
