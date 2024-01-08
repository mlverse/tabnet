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

test_that("explain provides interprestability as soon as there are 5 checkpoints", {

  # formula, pretrain
  tabnet_pretrain <- tabnet_pretrain(Sale_Price ~., data=small_ames, epochs = 3, valid_split=.2,
                                     checkpoint_epoch = 1,
                                     num_steps = 1, attention_width = 1, num_shared = 1, num_independent = 1,
                                     verbose = FALSE)
  expect_no_error(
    expl <- tabnet_explain(tabnet_pretrain, new_data = small_ames)
  )
  # not enough checkpoints
  expect_equal(expl$interprestability, NA_real_)
  # formula, pretrain continuation
  tabnet_pretrain <- tabnet_pretrain(Sale_Price ~., data=small_ames,
                                     epochs = 6, valid_split = 0.2, checkpoint_epoch = 1,
                                     num_steps = 1, attention_width = 1, num_shared = 1, num_independent = 1,
                                     verbose = FALSE)
  expect_no_error(
    expl <- tabnet_explain(tabnet_pretrain, new_data = small_ames)
  )
  # 5 checkpoints and more
  expect_gt(expl$interprestability, 0.5)

  # formula, fit
  tabnet_fit <- tabnet_fit(Sale_Price ~., data=small_ames, tabnet_model = tabnet_pretrain, epochs = 3,
                           num_steps = 1, attention_width = 1, num_shared = 1, num_independent = 1,
                           checkpoint_epoch = 1, verbose = FALSE)
  expect_no_error(
    expl <- tabnet_explain(tabnet_fit, new_data = small_ames)
  )
  # not enough checkpoints
  expect_equal(expl$interprestability, NA_real_)
  # formula, fit continuation
  tabnet_fit <- tabnet_fit(Sale_Price ~., data=small_ames, tabnet_model = tabnet_fit, epochs = 3,
                           num_steps = 1, attention_width = 1, num_shared = 1, num_independent = 1,
                           checkpoint_epoch = 1, verbose = FALSE)
  expect_no_error(
    expl <- tabnet_explain(tabnet_fit, new_data = small_ames)
  )
  # 5 checkpoints and more
  expect_gt(expl$interprestability, 0.5)

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

