# as per https://github.com/mlverse/torch/blob/main/tests/testthat/helper-tensor.R
Sys.setenv(KMP_DUPLICATE_LIB_OK = TRUE)
# torch_zeros(1, names="hello") # trigger warning about named tensors

skip_if_not_test_examples <- function() {
  if (Sys.getenv("TEST_EXAMPLES", unset = "0") != "1") {
    skip("Not testing examples/readme. Set the env var TEST_EXAMPLES = 1.")
  }
}

skip_if_cuda_not_available <- function() {
  if (!cuda_is_available()) {
    skip("A GPU is not available for testing.")
  }
}

skip_if_not_m1_mac <- function() {
  if (!grepl("darwin", R.version$os)) {
    skip("Not on MacOS")
  }

  if (R.version$arch != "aarch64") {
    skip("Not an M1 Mac")
  }
}

expect_equal_to_tensor <- function(object, expected, ...) {
  expect_equal(torch::as_array(object), torch::as_array(expected), ...)
}

expect_not_equal_to_tensor <- function(object, expected) {
  expect_false(isTRUE(all.equal(torch::as_array(object), torch::as_array(expected))))
}

expect_no_error <- function(object, ...) {
  expect_error(object, NA, ...)
}

expect_tensor <- function(object) {
  expect_true(torch:::is_torch_tensor(object))
  expect_no_error(torch::as_array(object$to(device = "cpu")))
}

expect_equal_to_r <- function(object, expected, ...) {
  expect_equal(torch::as_array(object$cpu()), expected, ...)
}

expect_tensor_shape <- function(object, expected) {
  expect_tensor(object)
  expect_equal(object$shape, expected)
}

expect_undefined_tensor <- function(object) {
  # TODO
}

expect_identical_modules <- function(object, expected) {
  expect_identical(
    attr(object, "module"),
    attr(expected, "module")
  )
}
