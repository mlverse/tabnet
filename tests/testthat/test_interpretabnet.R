test_that("mlp module works", {
  expect_no_error(
    att_tr <- tabnet:::attentive_transformer(10, 15, mlp_hidden_mults = c(4,2))
  )
  expect_tensor_shape(att_tr$parameters$mlp.mlp.0.weight, c(4,10))
  expect_tensor_shape(att_tr$parameters$mlp.mlp.2.weight, c(2,4))
  expect_tensor_shape(att_tr$parameters$mlp.mlp.4.weight, c(15, 2))

  expect_no_error(
    encoder <- tabnet:::tabnet_encoder(10, 15, , mlp_hidden_mults = c(4,2))
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

