test_that("nn_unsupervised_loss is working as expected", {
  
  unsup_loss <- tabnet:::nn_unsupervised_loss()
  
  # the poor-guy expect_r6_class(x, class)
  expect_true(all(c("nn_weighted_loss","nn_loss","nn_module") %in% class(unsup_loss)))
  
  y_pred <- torch::torch_rand(3,5, requires_grad = TRUE)
  embedded_x <- torch::torch_rand(3,5)
  obfuscation_mask <- torch::torch_bernoulli(embedded_x, p = 0.5)
  output <- unsup_loss(y_pred, embedded_x, obfuscation_mask)
  output$backward()
  
  expect_tensor(output)
  expect_equal_to_r(output >= 0, TRUE) 
  expect_false(rlang::is_null(output$grad_fn))
  expect_equal(output$dim(), 0)
})


test_that("nn_aum_loss works as expected with 1-dim label", {
  
  aum_loss <- tabnet::nn_aum_loss()
  
  # the poor-guy expect_r6_class(x, class)
  expect_true(all(c("nn_mse_loss","nn_loss","nn_module") %in% class(aum_loss)))
  
  # 1-dim label
  label_tensor <- torch::torch_tensor(attrition$Attrition)
  pred_tensor <- torch::torch_rand(label_tensor$shape, requires_grad = TRUE) 
  output <- aum_loss(pred_tensor, label_tensor)
  output$backward()
  
  expect_tensor(output)
  expect_equal_to_r(output >= 0, TRUE) 
  expect_false(rlang::is_null(output$grad_fn))
  expect_equal(output$dim(), 0)
  
})


test_that("nn_aum_loss works as expected with 2-dim label", {
  
  aum_loss <- tabnet::nn_aum_loss()
  label_tensor <- torch::torch_tensor(attrition$Attrition)$unsqueeze(-1)
  pred_tensor <- torch::torch_rand(label_tensor$shape, requires_grad = TRUE)
  output <- aum_loss(pred_tensor, label_tensor)
  output$backward()
  
  expect_tensor(output)
  expect_equal_to_r(output >= 0, TRUE) 
  expect_false(rlang::is_null(output$grad_fn))
  expect_equal(output$dim(), 0)
})


test_that("nn_aum_loss works as expected with {n, 2} shape prediction", {
  
  aum_loss <- tabnet::nn_aum_loss()
  label_tensor <- torch::torch_tensor(attrition$Attrition)
  pred_tensor <- torch::torch_rand(c(label_tensor$shape, 2), requires_grad = TRUE) 
  output <- aum_loss(pred_tensor, label_tensor)
  output$backward()
  
  
  expect_tensor(output)
  expect_equal_to_r(output >= 0, TRUE) 
  expect_false(rlang::is_null(output$grad_fn))
  expect_equal(output$dim(), 0)
  
})


test_that("get_constr_output handles basic 2D input with identity constraint", {
  m <- matrix(c(1, 2, 
                3, 4), nrow = 2, ncol = 2)
  x <- torch_tensor(m, dtype = torch::torch_float32())
  R <- torch::torch_eye(2, dtype = torch::torch_float32())
  result <-get_constr_output(x, R)
  expect_tensor(result)
  expect_tensor_shape(result, c(2, 2))
  expect_equal_to_r(result, m)
})

test_that("get_constr_output applies hierarchy constraint correctly", {
  x <- torch_tensor(matrix(c(1, 5, 
                             3, 2), nrow = 2, ncol = 2, byrow = TRUE), dtype = torch::torch_float64())
  R <- torch_tensor(matrix(c(1, 1, 
                             0, 1), nrow = 2, ncol = 2, byrow = TRUE), dtype = torch::torch_float64())
  result <-get_constr_output(x, R)
  expect_tensor_shape(result, c(2, 2))
  expected <- matrix(c(5, 5, 3, 2), nrow = 2, ncol = 2, byrow = TRUE)
  expect_equal_to_r(result, expected, tolerance = 1e-6)
})

test_that("get_constr_output preserves input dtype", {
  x_f32 <- torch_tensor(matrix(1:4, nrow = 2), dtype = torch::torch_float32())
  x_f64 <- torch_tensor(matrix(1:4, nrow = 2), dtype = torch::torch_float64())
  R <- torch::torch_eye(2)
  expect_tensor_dtype(get_constr_output(x_f32, R), torch::torch_float64())
  expect_tensor_dtype(get_constr_output(x_f64, R), torch::torch_float64())
})

test_that("get_constr_output handles batch dimension correctly", {
  x <- torch_tensor(matrix(1:12, nrow = 3, ncol = 4))
  R <- torch_tensor(matrix(c(1, 1, 0, 0, 
                             1, 1, 0, 0, 
                             0, 0, 1, 1, 
                             0, 0, 1, 1), nrow = 4, ncol = 4, byrow = TRUE))
  result <-get_constr_output(x, R)
  expect_tensor_shape(result, c(3, 4))
  
  for (i in 1:3) {
    row_result <- as_array(result[i, ])
    max_grp1 <- max(as_array(x[i, 1:2]))
    max_grp2 <- max(as_array(x[i, 3:4]))
    
    expect_equal(row_result[1:2], rep(max_grp1, 2), tolerance = 1e-6)
    expect_equal(row_result[3:4], rep(max_grp2, 2), tolerance = 1e-6)
  }
})
test_that("get_constr_output works with single sample", {
  x <- torch_tensor(matrix(c(2, 1, 4, 3), nrow = 1, ncol = 4, byrow = TRUE))
  R <- torch_tensor(matrix(c(1, 1, 0, 0, 
                             1, 1, 0, 0, 
                             0, 0, 1, 1, 
                             0, 0, 1, 1), nrow = 4, ncol = 4, byrow = TRUE))
  result <-get_constr_output(x, R)
  expect_tensor_shape(result, c(1, 4))
  expected <- matrix(c(2, 2, 4, 4), nrow = 1, byrow = TRUE)
  expect_equal_to_r(result, expected)
})

test_that("get_constr_output handles all-zeros constraint matrix", {
  x <- torch_tensor(matrix(1:6, nrow = 2, ncol = 3))
  R <- torch::torch_zeros(c(3, 3))
  result <-get_constr_output(x, R)
  expect_tensor_shape(result, c(2, 3))
  expect_equal_to_r(result, matrix(0, nrow = 2, ncol = 3))
})

test_that("get_constr_output handles all-ones constraint matrix", {
  x <- torch_tensor(matrix(c(1, 5, 3, 
                             2, 4, 6), nrow = 2, ncol = 3, byrow = TRUE))
  R <- torch::torch_ones(c(3, 3))
  result <-get_constr_output(x, R)
  expect_tensor_shape(result, c(2, 3))
  # Each row is filled with its own row-wise maximum
  expected <- matrix(c(5, 5, 5, 
                       6, 6, 6), nrow = 2, ncol = 3, byrow = TRUE)
  expect_equal_to_r(result, expected, tolerance = 1e-6)
})

test_that("get_constr_output throws error for dimension mismatch", {
  x <- torch_tensor(matrix(1:4, nrow = 2, ncol = 2))
  R <- torch::torch_eye(3)
  expect_error(get_constr_output(x, R), "must match the existing size")
})

test_that("get_constr_output throws error for non-2D R", {
  x <- torch_tensor(matrix(1:4, nrow = 2, ncol = 2))
  R <- torch_tensor(array(1:8, dim = c(1, 2, 2, 2)))
  expect_error(get_constr_output(x, R), "dimension")
})

test_that("get_constr_output handles negative values correctly", {
  x <- torch_tensor(matrix(c(-5, -1, 
                             -3, -2), nrow = 2, ncol = 2, byrow = TRUE))
  R <- torch_tensor(matrix(c(1, 1, 
                             0, 1), nrow = 2, ncol = 2, byrow = TRUE))
  result <- get_constr_output(x, R)
  expected <- matrix(c(-1, 0, 
                       -2, 0), nrow = 2, ncol = 2, byrow = TRUE)
  expect_equal_to_r(result, expected)
})
