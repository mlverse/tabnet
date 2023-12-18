processed_feat <- torch::torch_rand(c(32, 32))
prior <- torch::torch_bernoulli(processed_feat, 0.2)

test_that("mlp module works", {

  # init
  expect_no_error(
    att_tr <- tabnet:::attentive_transformer(32, 16, mlp_hidden_mults = c(4,2))
  )

  # init
  expect_no_error(
    encoder <- tabnet:::tabnet_encoder(32, 16, mlp_hidden_mults = c(4,2))
  )

  # compute
  expect_no_error(
    encoder(processed_feat, prior)
  )
})

test_that("mlp module works with non default activation function", {
  # init
  expect_no_error(
    att_tr <- tabnet:::attentive_transformer(32, 16, mlp_hidden_mults = c(4,2),  mlp_act = torch::nn_leaky_relu())
  )
  expect_tensor_shape(att_tr$parameters$mlp.mlp.0.weight, c(8,32))
  expect_tensor_shape(att_tr$parameters$mlp.mlp.2.weight, c(4,8))
  expect_tensor_shape(att_tr$parameters$mlp.mlp.4.weight, c(16, 4))

  # init
  expect_no_error(
    encoder <- tabnet:::tabnet_encoder(32, 16, mlp_hidden_mults = c(4,2),  mlp_act = torch::nn_leaky_relu())
  )
  # compute
  expect_no_error(
    encoder(processed_feat, prior)
  )
})

test_that("encoder module works with non default activation function", {
  # init
  expect_no_error(
    encoder <- tabnet:::tabnet_encoder(32, 16, mlp_hidden_mults = c(4,2),
                                       mlp_act = torch::nn_leaky_relu(),
                                       encoder_act = torch::nnf_leaky_relu )
  )
  expect_tensor_shape(encoder$parameters$att_transformers.0.mlp.mlp.0.weight, c(16,8))
  expect_tensor_shape(encoder$parameters$att_transformers.0.mlp.mlp.2.weight, c(8,16))
  expect_tensor_shape(encoder$parameters$att_transformers.0.mlp.mlp.4.weight, c(32,8))
  expect_tensor_shape(encoder$parameters$att_transformers.1.mlp.mlp.0.weight, c(16,8))
  expect_tensor_shape(encoder$parameters$att_transformers.1.mlp.mlp.2.weight, c(8,16))
  expect_tensor_shape(encoder$parameters$att_transformers.1.mlp.mlp.4.weight, c(32,8))
  # compute
  expect_no_error(
    c(res, M_loss, steps_output) %<-% encoder(processed_feat, prior)
  )
  expect_tensor_shape(res, c(32,8))
  expect_tensor_shape(steps_output[[1]], c(32,8))
  expect_tensor_shape(steps_output[[2]], c(32,8))
  expect_tensor_shape(steps_output[[3]], c(32,8))
})

test_that("encoder module works with non mBwLU activation function", {
  # init
  expect_no_error(
    encoder <- tabnet:::tabnet_encoder(32, 16, mlp_hidden_mults = c(4,2),
                                       mlp_act = torch::nn_leaky_relu(),
                                       encoder_act = nn_mb_wlu() )
  )
  expect_no_error(
    encoder <- tabnet:::tabnet_encoder(32, 16, mlp_hidden_mults = c(4,2),
                                       mlp_act = torch::nn_leaky_relu(),
                                       encoder_act = nn_mb_wlu(alpha = 0.5, beta = 0.3, gamma = 0.2) )
  )
  expect_tensor_shape(encoder$parameters$att_transformers.0.mlp.mlp.0.weight, c(16,8))
  expect_tensor_shape(encoder$parameters$att_transformers.0.mlp.mlp.2.weight, c(8,16))
  expect_tensor_shape(encoder$parameters$att_transformers.0.mlp.mlp.4.weight, c(32,8))
  expect_tensor_shape(encoder$parameters$att_transformers.1.mlp.mlp.0.weight, c(16,8))
  expect_tensor_shape(encoder$parameters$att_transformers.1.mlp.mlp.2.weight, c(8,16))
  expect_tensor_shape(encoder$parameters$att_transformers.1.mlp.mlp.4.weight, c(32,8))
  # compute
  expect_no_error(
    c(res, M_loss, steps_output) %<-% encoder(processed_feat, prior)
  )
  expect_tensor_shape(res, c(32,8))
  expect_tensor_shape(steps_output[[1]], c(32,8))
  expect_tensor_shape(steps_output[[2]], c(32,8))
  expect_tensor_shape(steps_output[[3]], c(32,8))
})




