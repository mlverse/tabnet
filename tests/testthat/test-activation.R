test_that("multibranch_weighted_linear_unit activation works", {
  mb_wlu <- nn_mb_wlu()
  input <- torch::torch_tensor(c(-1.0, 0.0, 1.0))
  expected_output <- torch::torch_tensor(c(-0.48306063, 0.0, 0.94621176))

  expect_equal_to_tensor(mb_wlu(input), expected_output)

})

test_that("multibranch_weighted_linear_unit correctly prevent weight not being a tensor", {
  expect_error(mb_wlu <- nn_mb_wlu( weight = 0.25),
               regexp = "must be a torch_tensor")
})

test_that("multibranch_weighted_linear_unit correctly prevent weight not being on the same device", {
  skip_if_not(torch::backends_openmp_is_available())
  weight <- torch::torch_tensor(0.25)$to(device = "cpu")
  z <- torch::torch_randr(c(2,2))$to(device = "openmp")

  expect_no_error(mb_wlu <- nn_mb_wlu( weight = weight))
  expect_error(mb_wlu(z),
               regexp = "reside on the same device")
})

