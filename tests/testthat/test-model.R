expect_tensor_shape <- function(object, expected) {
  expect_true(torch:::is_torch_tensor(object))
  expect_error(torch::as_array(object$to(device = "cpu")), NA)
  expect_equal(object$shape, expected)
}

if (torch::cuda_is_available()) {
  device <- "cuda"
} else {
  device <- "cpu"
}


test_that("resolve_data gives expected output through a dataloader", {
  data("ames", package = "modeldata")

  x <- ames[-which(names(ames) == "Sale_Price")]
  y <- ames$Sale_Price
  # dataset are R6 class and shall be instanciated
  train_ds <- torch::dataset(
    initialize = function() {},
    .getbatch = function(batch) {resolve_data(x[batch,], y[batch], device=device)},
    .length = function() {nrow(x)}
  )()
  # dataloader
  train_dl <- torch::dataloader(
    train_ds,
    batch_size = 2000 ,
    drop_last = TRUE,
    shuffle = FALSE #,
    # num_workers = 0L
  )
  expect_error(
    train_ds$.getbatch(batch = c(1:2)),
    NA
  )
  expect_error(
    coro::loop(for (batch in train_dl) {
      expect_tensor_shape(batch$x, c(2000, 73))
      expect_true(batch$x$dtype == torch::torch_float())
      expect_tensor_shape(batch$x_na_mask, c(2000, 73))
      expect_true(batch$x_na_mask$dtype == torch::torch_bool())
      expect_tensor_shape(batch$y, c(2000, 1))
      expect_true(batch$y$dtype == torch::torch_float())
      expect_tensor_shape(batch$cat_idx, 40)
      expect_true(batch$cat_idx$dtype == torch::torch_long())
      expect_tensor_shape(batch$output_dim, 1)
      expect_true(batch$cat_idx$dtype == torch::torch_long())
      expect_tensor_shape(batch$input_dim, 1)
      expect_true(batch$input_dim$dtype == torch::torch_long())
      expect_tensor_shape(batch$cat_dims, 40)
      expect_true(batch$cat_dims$dtype == torch::torch_long())

    }),
    NA
  )

})

test_that("resolve_data gives expected output without nominal variables", {
  n <- 1000
  x <- data.frame(
    x = rnorm(n),
    y = rnorm(n),
    z = rnorm(n)
  )

  y <- x$x
  # dataset are R6 class and shall be instanciated
  train_ds <- torch::dataset(
    initialize = function() {},
    .getbatch = function(batch) {resolve_data(x[batch,], y[batch], device=device)},
    .length = function() {nrow(x)}
  )()
  # dataloader
  train_dl <- torch::dataloader(
    train_ds,
    batch_size = 2000 ,
    drop_last = TRUE,
    shuffle = FALSE #,
    # num_workers = 0L
  )

  expect_error(
    train_ds$.getbatch(batch = c(1:2)),
    NA
  )
  expect_error(
    coro::loop(for (batch in train_dl) {
      expect_tensor_shape(batch$x, c(2000, 3))
      expect_true(batch$x$dtype == torch::torch_float())
      expect_tensor_shape(batch$x_na_mask, c(2000, 3))
      expect_true(batch$x_na_mask$dtype == torch::torch_bool())
      expect_tensor_shape(batch$y, c(2000, 1))
      expect_true(batch$y$dtype == torch::torch_float())
      expect_tensor_shape(batch$cat_idx, 0)
      expect_true(batch$cat_idx$dtype == torch::torch_long())
      expect_tensor_shape(batch$output_dim, 1)
      expect_true(batch$cat_idx$dtype == torch::torch_long())
      expect_tensor_shape(batch$input_dim, 1)
      expect_true(batch$input_dim$dtype == torch::torch_long())
      expect_tensor_shape(batch$cat_dims, 0)
      expect_true(batch$cat_dims$dtype == torch::torch_long())

    }),
    NA
  )

})
