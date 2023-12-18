test_that("multibranch_weighted_linear_unit nn_module works", {
  mb_wlu <- nn_mb_wlu()
  input <- torch::torch_tensor(c(-1.0, 0.0, 1.0))
  expected_output <- torch::torch_tensor(c(-0.48306063, 0.0, 0.94621176))

  expect_equal_to_tensor(mb_wlu(input), expected_output)

})

