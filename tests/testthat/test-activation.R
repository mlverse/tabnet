test_that("multibranch_weighted_linear_unit allows weight not to be a tensor", {
  expect_no_error(mb_wlu <- nn_mb_wlu( weight = 0.25))
})

test_that("multibranch_weighted_linear_unit activation works", {
  mb_wlu <- nn_mb_wlu()
  input <- torch::torch_tensor(c(-1.0, 0.0, 1.0))
  expected_output <- torch::torch_tensor(c(-0.48306063, 0.0, 0.94621176))

  expect_equal_to_tensor(mb_wlu(input), expected_output)

  mb_wlu <- nn_mb_wlu(weight = 0.025)
  input <- torch::torch_tensor(c(-1.0, 0.0, 1.0))
  expected_output <- torch::torch_tensor(c(-0.4380606, 0.0, 0.94621176))

  expect_equal_to_tensor(mb_wlu(input), expected_output)

  mb_wlu <- nn_mb_wlu(alpha = 0.1, beta = 1.2, gamma = 0.7)
  input <- torch::torch_tensor(c(-1.0, 0.0, 1.0))
  expected_output <- torch::torch_tensor(c(-0.5514711, 0.0, 1.8117411))

  expect_equal_to_tensor(mb_wlu(input), expected_output)

})
