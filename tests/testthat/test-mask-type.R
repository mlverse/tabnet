
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


test_that("fit works with entmax mask-type", {
  
  rec <- recipe(EnvironmentSatisfaction ~ ., data = attrition[ids, ]) %>%
    step_normalize(all_numeric(), -all_outcomes())
  
  expect_no_error(
    tabnet_fit(rec, attrition, epochs = 1, valid_split = 0.25, verbose = TRUE,
               config = tabnet_config( mask_type = "entmax"))
  )
})


test_that("sparsemax_15_function works as a proper autograd", {
  
  input = torch::torch_rand(10,2, requires_grad = TRUE)

  expect_no_error(
    output <- tabnet:::sparsemax_function(input, 2L, 3)
  )
  expect_no_error(
    output$backward
  )
})

test_that("entmax_15_function works as a proper autograd", {
  
  input = torch::torch_rand(10,2, requires_grad = TRUE)

  expect_no_error(
    output <- tabnet:::entmax_15_function(input, 2L, 3)
  )
  expect_no_error(
    output$backward
  )
})


test_that("fit works with sparsemax15 mask-type", {
  
  rec <- recipe(EnvironmentSatisfaction ~ ., data = attrition[ids, ]) %>%
    step_normalize(all_numeric(), -all_outcomes())
  
  expect_no_error(
    tabnet_fit(rec, attrition, epochs = 1, valid_split = 0.25, verbose = TRUE,
               config = tabnet_config( mask_type = "sparsemax15"))
  )
  expect_no_error(
    tabnet_fit(rec, attrition, epochs = 1, verbose = TRUE,
               config = tabnet_config( mask_type = "sparsemax15", mask_topk = 12))
  )
})

test_that("fit works with entmax15 mask-type", {
  
  rec <- recipe(EnvironmentSatisfaction ~ ., data = attrition[ids, ]) %>%
    step_normalize(all_numeric(), -all_outcomes())
  
  expect_no_error(
    tabnet_fit(rec, attrition, epochs = 1, valid_split = 0.25, verbose = TRUE,
               config = tabnet_config( mask_type = "entmax15"))
  )
  expect_no_error(
    tabnet_fit(rec, attrition, epochs = 1, verbose = TRUE,
               config = tabnet_config( mask_type = "entmax15", mask_topk = 12))
  )
})
