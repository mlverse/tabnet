
test_that(".sparsemax_threshold_and_support works as expected with default values", {
  input <- torch::torch_randn(10, 5)
  expect_no_error(
    result <- tabnet:::.sparsemax_threshold_and_support(input)
  )
  expect_type(result, "list")
  expect_length(result, 2)

  tau <- result[[1]]
  support_size <- result[[2]]
  expect_tensor(tau)
  expect_tensor_shape(tau, c(input$shape[1], 1))
  expect_tensor(support_size)
})


test_that(".sparsemax_threshold_and_support works as expected with k < input$size(dim)", {
  input <- torch::torch_randn(10, 5)
  dim <- 1L
  k <- 3
  expect_no_error(
    result <- tabnet:::.sparsemax_threshold_and_support(input, dim, k)
  )
  expect_type(result, "list")
  expect_length(result, 2)

  tau <- result[[1]]
  support_size <- result[[2]]
  expect_tensor(tau)
  expect_tensor_shape(tau, c(1, input$shape[2]))
  expect_tensor(support_size)
  expect_tensor_shape(support_size, c(1, input$shape[2]))

})

test_that(".sparsemax_threshold_and_support works as expected with k >= input$size(dim)", {
  input <- torch::torch_randn(10, 5)
  dim <- -2L
  k <- 7
  expect_no_error(
    result <- tabnet:::.sparsemax_threshold_and_support(input, dim, k)
  )
  expect_type(result, "list")
  expect_length(result, 2)

  tau <- result[[1]]
  support_size <- result[[2]]
  expect_tensor(tau)
  expect_tensor_shape(tau, c(1, input$shape[2]))
  expect_tensor(support_size)
  expect_tensor_shape(support_size, c(1, input$shape[2]))
})



test_that(".entmax_threshold_and_support works as expected with default values", {
  input <- torch::torch_randn(10, 5)
  expect_no_error(
    result <- tabnet:::.entmax_threshold_and_support(input)
  )
  expect_type(result, "list")
  expect_length(result, 2)

  tau_star <- result[[1]]
  support_size <- result[[2]]
  expect_tensor(tau_star)
  expect_tensor_shape(tau_star, c(input$shape[1], 1))
  expect_tensor(support_size)
  expect_tensor_shape(support_size, c(input$shape[1], 1))
})


test_that(".entmax_threshold_and_support works as expected with k < input$size(dim)", {
  input <- torch::torch_randn(10, 5)
  dim <- 1L
  k <- 3
  expect_no_error(
    result <- tabnet:::.entmax_threshold_and_support(input, dim, k)
  )
  expect_type(result, "list")
  expect_length(result, 2)

  tau_star <- result[[1]]
  support_size <- result[[2]]
  expect_tensor(tau_star)
  expect_tensor_shape(tau_star, c(1, input$shape[2]))
  expect_tensor(support_size)
  expect_tensor_shape(support_size, c(1, input$shape[2]))
})


test_that(".entmax_threshold_and_support works as expected with k >= input$size(dim)", {
  input <- torch::torch_randn(10, 5)
  dim <- 2L
  k <- 12
  expect_no_error(
    result <- tabnet:::.entmax_threshold_and_support(input, dim, k)
  )
  expect_type(result, "list")
  expect_length(result, 2)

  tau_star <- result[[1]]
  support_size <- result[[2]]
  expect_tensor(tau_star)
  expect_tensor_shape(tau_star, c(input$shape[1], 1))
  expect_tensor(support_size)
  expect_tensor_shape(support_size, c(input$shape[1], 1))
})
