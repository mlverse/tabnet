test_that("mlp module works", {
  expect_no_error(
    att_tr <- tabnet:::attentive_transformer(10, 15, mlp_hidden_mults = c(4,2))
  )
  expect_tensor_shape(att_tr$parameters$mlp.mlp.0.weight, c(4,10))
  expect_tensor_shape(att_tr$parameters$mlp.mlp.2.weight, c(2,4))
  expect_tensor_shape(att_tr$parameters$mlp.mlp.4.weight, c(15, 2))

  expect_no_error(
    encoder <- tabnet:::tabnet_encoder(10, 15, mlp_hidden_mults = c(4,2))
  )
  expect_tensor_shape(encoder$parameters$att_transformers.0.mlp.mlp.0.weight, c(4,8))
  expect_tensor_shape(encoder$parameters$att_transformers.0.mlp.mlp.2.weight, c(2,4))
  expect_tensor_shape(encoder$parameters$att_transformers.0.mlp.mlp.4.weight, c(10,2))
  expect_tensor_shape(encoder$parameters$att_transformers.1.mlp.mlp.0.weight, c(4,8))
  expect_tensor_shape(encoder$parameters$att_transformers.1.mlp.mlp.2.weight, c(2,4))
  expect_tensor_shape(encoder$parameters$att_transformers.1.mlp.mlp.4.weight, c(10,2))
})

test_that("mlp module works with non default activation function", {
  expect_no_error(
    att_tr <- tabnet:::attentive_transformer(10, 15, mlp_hidden_mults = c(4,2), mlp_act = torch::nn_leaky_relu())
  )
  expect_tensor_shape(att_tr$parameters$mlp.mlp.0.weight, c(4,10))
  expect_tensor_shape(att_tr$parameters$mlp.mlp.2.weight, c(2,4))
  expect_tensor_shape(att_tr$parameters$mlp.mlp.4.weight, c(15, 2))

  expect_no_error(
    encoder <- tabnet:::tabnet_encoder(10, 15, mlp_hidden_mults = c(4,2), mlp_act = torch::nn_leaky_relu())
  )
  expect_tensor_shape(encoder$parameters$att_transformers.0.mlp.mlp.0.weight, c(4,8))
  expect_tensor_shape(encoder$parameters$att_transformers.0.mlp.mlp.2.weight, c(2,4))
  expect_tensor_shape(encoder$parameters$att_transformers.0.mlp.mlp.4.weight, c(10,2))
  expect_tensor_shape(encoder$parameters$att_transformers.1.mlp.mlp.0.weight, c(4,8))
  expect_tensor_shape(encoder$parameters$att_transformers.1.mlp.mlp.2.weight, c(2,4))
  expect_tensor_shape(encoder$parameters$att_transformers.1.mlp.mlp.4.weight, c(10,2))
})

test_that("mlp module does not bring regression to tabnet", {
  expect_no_error(
    att_tr <- tabnet:::attentive_transformer(10, 15, mlp_hidden_mults = NULL)
  )
  expect_tensor_shape(att_tr$parameters$mlp.mlp.weight, c(15,10))

  expect_no_error(
    encoder <- tabnet:::tabnet_encoder(10, 15)
  )
  expect_tensor_shape(encoder$parameters$att_transformers.0.mlp.mlp.weight, c(10,8))
  expect_tensor_shape(encoder$parameters$att_transformers.1.mlp.mlp.weight, c(10,8))
})

test_that("`%!=%` works for R configuration parameters as for torch::nn_modules", {
  a <- torch::nn_relu()
  b <- torch::nn_relu6()
  expect_true(tabnet:::`%!=%`(a, b))
  a <- torch::nn_elu()
  b <- torch::nn_elu(alpha = 0.8)
  expect_true(tabnet:::`%!=%`(a, b))
  c <- list(param_a = 1, param_b = c(4,2))
  d <- list(param_a = 2, param_b = c(4,2))
  e <- list(param_a = 1, param_b = c(4,3))
  f <- list(param_a = 1, param_b = c(4,2,1))
  expect_true(tabnet:::`%!=%`(b, c))
  expect_true(tabnet:::`%!=%`(c, d))
  expect_true(tabnet:::`%!=%`(c, e))
  expect_true(tabnet:::`%!=%`(c, f))
  f <- tabnet_config()
  g <- interpretabnet_config()
  expect_true(tabnet:::`%!=%`(f, g))

})

