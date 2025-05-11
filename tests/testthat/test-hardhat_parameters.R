test_that("errors when using an argument that do not exist", {

  expect_error(
    fit <- tabnet_fit(x, y, epochsas = 1),
    "unused argument"
  )

  expect_error(
    fit <- tabnet_fit(x, y, config = tabnet_config(epochsas = 1)),
    "unused argument"
  )

})

test_that("merging parameters from config and dots works", {
  
  # we can merge non-atomic values on different variables (no comparison)
  expect_no_error(
    merged_conf <- tabnet:::merge_config_and_dots(
      config = tabnet_config(optimizer = torch::optim_adamw),
      loss = torch::nn_bce_loss)
  )
  expect_identical_modules(merged_conf$optimizer, torch::optim_adamw)
  expect_identical_modules(merged_conf$loss, torch::nn_bce_loss)
  
  # we can merge non-atomic values on different variables while resolving optimizer
  expect_no_error(
    merged_conf <- tabnet:::merge_config_and_dots(
      config = tabnet_config(optimizer = "adam"),
      loss = torch::nn_bce_loss)
  )
  if(tabnet:::torch_has_optim_ignite()) {
    expect_identical_modules(merged_conf$optimizer, torch::optim_ignite_adam)
  } else {
    expect_identical_modules(merged_conf$optimizer, torch::optim_adam)
  }
  expect_identical_modules(merged_conf$loss, torch::nn_bce_loss)
  
  # ... value wins over tabnet_config() value
  expect_no_error(
    merged_conf <- tabnet:::merge_config_and_dots(
      config = tabnet_config(batch_size = 200),
      batch_size = 400)
  )
  expect_identical(merged_conf$batch_size, 400)
  
  # ... value wins over tabnet_config() value for non-atomic parameter
  expect_no_error(
    merged_conf <- tabnet:::merge_config_and_dots(
      config = tabnet_config(loss = torch::nn_cross_entropy_loss),
      loss = torch::nn_bce_loss)
  )
  expect_identical_modules(merged_conf$loss, torch::nn_bce_loss)
  
  # NULL value get replaced and optimizer is resolved even lately
  expect_no_error(
    merged_conf <- tabnet:::merge_config_and_dots(
      config = tabnet_config(loss = NULL, device = "cuda"),
      loss = torch::nn_bce_loss, optimizer = "adam")
  )
  if(tabnet:::torch_has_optim_ignite()) {
    expect_identical_modules(merged_conf$optimizer, torch::optim_ignite_adam)
  } else {
    expect_identical_modules(merged_conf$optimizer, torch::optim_adam)
  }
  expect_identical_modules(merged_conf$loss, torch::nn_bce_loss)

  
})

test_that("pretrain and fit both work with early stopping", {

  expect_message(
    pretrain <- tabnet_pretrain(attrix, attriy, epochs = 100, valid_split = 0.5, verbose=TRUE,
                                early_stopping_tolerance=1e-7, early_stopping_patience=3, learn_rate = 0.2),
    "Early-stopping at epoch"
  )
  expect_lt(length(pretrain$fit$metrics),100)

  expect_message(
    fit <- tabnet_fit(attrix, attriy, epochs = 100, valid_split = 0.5, verbose=TRUE,
                      early_stopping_tolerance=1e-7, early_stopping_patience=3, learn_rate = 0.2),
    "Early-stopping at epoch"
  )
  expect_lt(length(fit$fit$metrics),100)

})

test_that("early stopping works wo validation split", {

  # tabnet_pretrain
  expect_message(
    pretrain <- tabnet_pretrain(attrix, attriy, epochs = 100, verbose=TRUE,
                                early_stopping_monitor="train_loss",
                                early_stopping_tolerance=1e-7, early_stopping_patience=3, learn_rate = 0.2),
    "Early-stopping at epoch"
  )
  expect_lt(length(pretrain$fit$metrics),100)

  expect_error(
    tabnet_pretrain(attrix, attriy, epochs = 100, verbose=TRUE,
                                early_stopping_monitor="cross_validation_loss",
                                early_stopping_tolerance=1e-7, early_stopping_patience=3, learn_rate = 0.2),
    regexp = "not a valid early-stopping metric to monitor"
  )

  # tabnet_fit
  expect_message(
    fit <- tabnet_fit(attrix, attriy, epochs = 200, verbose=TRUE,
                      early_stopping_monitor="train_loss",
                      early_stopping_tolerance=1e-7, early_stopping_patience=3, learn_rate = 0.3),
    "Early-stopping at epoch"
  )
  expect_lt(length(fit$fit$metrics),200)

  expect_error(
    tabnet_fit(attrix, attriy, epochs = 200, verbose=TRUE,
                      early_stopping_monitor="cross_validation_loss",
                      early_stopping_tolerance=1e-7, early_stopping_patience=3, learn_rate = 0.2),
    regexp = "not a valid early-stopping metric to monitor"
  )

})

test_that("configuration with categorical_embedding_dimension vector works", {

  config <- tabnet_config(cat_emb_dim=c(1,1,2,2,1,1,1,2,1,1,1,2,2,2))

  expect_no_error(
    fit <- tabnet_fit(attrix, attriy, epochs = 1, valid_split = 0.2, config=config)
  )
})

test_that("explicit error message when categorical embedding dimension vector has wrong size", {

  config <- tabnet_config(cat_emb_dim=c(1,1,2,2))

  expect_error(
    fit <- tabnet_fit(attrix, attriy, epochs = 1, valid_split = 0.2, config=config),
    regexp = "number of categorical predictors"
  )
})

test_that("step scheduler works", {

  expect_no_error(
    fit <- tabnet_fit(x, y, epochs = 3, lr_scheduler = "step",
                      lr_decay = 0.1, step_size = 1)
  )

  sc_fn <- function(optimizer) {
    torch::lr_step(optimizer, step_size = 1, gamma = 0.1)
  }

  expect_no_error(
    fit <- tabnet_fit(x, y, epochs = 3, lr_scheduler = sc_fn,
                      lr_decay = 0.1, step_size = 1)
  )

})

test_that("configuring optimizer works", {

  expect_no_error(
    fit <- tabnet_fit(x, y, epochs = 3, config = tabnet_config(optimizer = "adam"))
  )

  expect_no_error(
    fit <- tabnet_fit(x, y, epochs = 3, optimizer = torch::optim_adamw)
  )

  expect_error(
    fit <- tabnet_fit(x, y, epochs = 3, optimizer = "adamw"),
    "Currently only \"adam\" is supported"
  )

})

test_that("reduce_on_plateau scheduler works", {

  expect_no_error(
    fit <- tabnet_fit(x, y, epochs = 3, lr_scheduler = "reduce_on_plateau",
                      lr_decay = 0.1, step_size = 1)
  )

  expect_error(
    fit <- tabnet_fit(x, y, epochs = 3, lr_scheduler = "multiplicative",
                      lr_decay = 0.1, step_size = 1),
    "only the \"step\" and \"reduce_on_plateau\" scheduler"
  )

  sc_fn <- function(optimizer) {
    torch::lr_reduce_on_plateau(optimizer, factor = 0.1, patience = 10)
  }

  expect_no_error(
    fit <- tabnet_fit(x, y, epochs = 3, lr_scheduler = sc_fn,
                      lr_decay = 0.1, step_size = 1)
  )

})

test_that("fit uses config parameters mix from config= and ...", {

  rec <- recipe(EnvironmentSatisfaction ~ ., data = attrition[ids, ]) %>%
    step_normalize(all_numeric(), -all_outcomes())
  fit <- tabnet_fit(rec, attrition, epochs = 1, valid_split = 0.25, verbose = TRUE,
                    config = tabnet_config(decision_width=3, attention_width=5, cat_emb_dim = 2))
  expect_no_error(
    predict(fit, attrition)
  )

  expect_equal(fit$fit$config$verbose, TRUE)
  expect_equal(fit$fit$config$valid_split, 0.25)
  expect_equal(fit$fit$config$n_d, 3)
  expect_equal(fit$fit$config$n_a, 5)

})


test_that("fit raise an error with non-supported mask-type", {

  rec <- recipe(EnvironmentSatisfaction ~ ., data = attrition[ids, ]) %>%
    step_normalize(all_numeric(), -all_outcomes())
  expect_error(
    tabnet_fit(rec, attrition, epochs = 1, valid_split = 0.25, verbose = TRUE,
                      config = tabnet_config( mask_type="max_entropy")),
    regexp = "either \"sparsemax\", \"sparsemax15\", \"entmax\" or \"entmax15\" as"
  )

})

test_that("config$loss=`auto` adapt to recipe outcome str()", {

  testthat::skip_on_ci()

  # nominal outcome
  rec <- recipe(EnvironmentSatisfaction ~ ., data = attrition[ids, ]) %>%
    step_normalize(all_numeric(), -all_outcomes())
  fit_auto <- tabnet_fit(rec, attrition[ids, ], epochs = 1, verbose = TRUE,
                      config = tabnet_config( loss="auto"))
  expect_equal(fit_auto$fit$config$loss_fn, torch::nn_cross_entropy_loss(), ignore_function_env = TRUE)

  # numerical outcome
  rec <- recipe(MonthlyIncome ~ ., data = attrition[ids, ]) %>%
    step_normalize(all_numeric(), -all_outcomes())
  fit_auto <- tabnet_fit(rec, attrition[ids, ], epochs = 1, verbose = TRUE,
                      config = tabnet_config( loss="auto"))
  expect_equal(fit_auto$fit$config$loss_fn, torch::nn_mse_loss(), ignore_function_env = TRUE)

})

test_that("config$loss not adapted to recipe outcome raise an explicit error", {

  testthat::skip_on_ci()

  # nominal outcome with numerical loss
  rec <- recipe(EnvironmentSatisfaction ~ ., data = attrition[ids, ]) %>%
    step_normalize(all_numeric(), -all_outcomes())
  expect_error(tabnet_fit(rec, attrition[ids, ], epochs = 1, verbose = TRUE,
                          config = tabnet_config( loss="mse")),
              regexp = "is not a valid loss for outcome of type"
  )
  # numerical outcome
  rec <- recipe(MonthlyIncome ~ ., data = attrition[ids, ]) %>%
    step_normalize(all_numeric(), -all_outcomes())
  expect_error(tabnet_fit(rec, attrition[ids, ], epochs = 1, verbose = TRUE,
                      config = tabnet_config( loss="cross_entropy")),
               regexp = "is not a valid loss for outcome of type"
  )
})


test_that("config$loss can be a function", {

  testthat::skip_on_ci()

  # nominal outcome loss
  rec <- recipe(EnvironmentSatisfaction ~ ., data = attrition[ids, ]) %>%
    step_normalize(all_numeric(), -all_outcomes())
  fit_auto <- tabnet_fit(rec, attrition[ids, ], epochs = 1, verbose = TRUE,
                      config = tabnet_config( loss = torch::nn_nll_loss()))
  expect_equal(fit_auto$fit$config$loss_fn, torch::nn_nll_loss(), ignore_function_env = TRUE)

  # numerical outcome loss
  rec <- recipe(MonthlyIncome ~ ., data = attrition[ids, ]) %>%
    step_normalize(all_numeric(), -all_outcomes())
  fit_auto <- tabnet_fit(rec, attrition[ids, ], epochs = 1, verbose = TRUE,
                      config = tabnet_config( loss = torch::nn_poisson_nll_loss()))
  expect_equal(fit_auto$fit$config$loss_fn, torch::nn_poisson_nll_loss(), ignore_function_env = TRUE)

})

