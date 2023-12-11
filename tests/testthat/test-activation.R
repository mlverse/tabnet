test_that("multibranch_weighted_linear_unit works", {
  mb_wlu <- nn_mbwlu()
  input <- torch_tensor(c(-1.0, 0.0, 1.0))
  expected_output <- torch_tensor(c(-0.26894142, 0.0, 0.73105858))

  expect_equal_to_tensor(mb_wlu(input), expected_output)

})

