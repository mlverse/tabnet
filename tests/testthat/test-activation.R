test_that("multibranch_weighted_linear_unit nn_module works", {
  mb_wlu <- nn_mbwlu()
  input <- torch::torch_tensor(c(-1.0, 0.0, 1.0))
  expected_output <- torch::torch_tensor(c(-0.4830606, 0.0, 0.9462118))

  expect_equal_to_tensor(mb_wlu(input), expected_output)

})

