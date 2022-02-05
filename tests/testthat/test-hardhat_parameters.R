test_that("errors when using an argument that do not exist", {

  data("ames", package = "modeldata")

  x <- ames[1:256,-which(names(ames) == "Sale_Price")]
  y <- ames$Sale_Price[1:256]

  expect_error(
    fit <- tabnet_fit(x, y, epochsas = 1),
    "unused argument"
  )

})

test_that("pretrain and fit both work with early stopping", {

  data("attrition", package = "modeldata")

  x <- attrition[1:256, -which(names(attrition) == "Attrition")]
  y <- attrition[1:256, "Attrition"]

  expect_message(
    pretrain <- tabnet_pretrain(x, y, epochs = 100, valid_split = 0.5, verbose=TRUE,
                                early_stopping_tolerance=0.001, early_stopping_patience=3, learn_rate = 0.2),
    "Early stopping at epoch"
  )
  expect_lt(length(pretrain$fit$metrics),50)

  expect_message(
    fit <- tabnet_fit(x, y, epochs = 100, valid_split = 0.5, verbose=TRUE,
                      early_stopping_tolerance=0.001, early_stopping_patience=3, learn_rate = 0.2),
    "Early stopping at epoch"
  )
  expect_lt(length(fit$fit$metrics),50)

})

test_that("early stopping works wo validation split", {

  data("attrition", package = "modeldata")

  x <- attrition[1:256, -which(names(attrition) == "Attrition")]
  y <- attrition[1:256, "Attrition"]

  # tabnet_pretrain
  expect_message(
    pretrain <- tabnet_pretrain(x, y, epochs = 100, verbose=TRUE,
                                early_stopping_monitor="train_loss",
                                early_stopping_tolerance=0.001, early_stopping_patience=3, learn_rate = 0.2),
    "Early stopping at epoch"
  )
  expect_lt(length(pretrain$fit$metrics),50)

  expect_error(
    pretrain <- tabnet_pretrain(x, y, epochs = 100, verbose=TRUE,
                                early_stopping_monitor="cross_validation_loss",
                                early_stopping_tolerance=0.001, early_stopping_patience=3, learn_rate = 0.2),
    regexp = "not a valid early stopping metric to monitor"
  )

  # tabnet_fit
  expect_message(
    fit <- tabnet_fit(x, y, epochs = 100, verbose=TRUE,
                      early_stopping_monitor="train_loss",
                      early_stopping_tolerance=0.001, early_stopping_patience=3, learn_rate = 0.2),
    "Early stopping at epoch"
  )
  expect_lt(length(fit$fit$metrics),50)

  expect_error(
    fit <- tabnet_fit(x, y, epochs = 100, verbose=TRUE,
                      early_stopping_monitor="cross_validation_loss",
                      early_stopping_tolerance=0.001, early_stopping_patience=3, learn_rate = 0.2),
    regexp = "not a valid early stopping metric to monitor"
  )

})


test_that("configuration with categorical_embedding_dimension vector works", {

  data("attrition", package = "modeldata")

  x <- attrition[1:256,-which(names(attrition) == "Attrition")]
  y <- attrition$Attrition[1:256]
  config <- tabnet_config(cat_emb_dim=c(1,1,2,2,1,1,1,2,1,1,1,2,2,2))

  expect_error(
    fit <- tabnet_fit(x, y, epochs = 1, valid_split = 0.2, config=config),
    regexp = NA
  )
})

test_that("explicit error message when categorical embedding dimension vector has wrong size", {

  data("attrition", package = "modeldata")

  x <- attrition[1:256,-which(names(attrition) == "Attrition")]
  y <- attrition$Attrition[1:256]
  config <- tabnet_config(cat_emb_dim=c(1,1,2,2))

  expect_error(
    fit <- tabnet_fit(x, y, epochs = 1, valid_split = 0.2, config=config),
    regexp = "number of categorical predictors"
  )
})

test_that("scheduler works", {

  data("ames", package = "modeldata")

  x <- ames[-which(names(ames) == "Sale_Price")]
  y <- ames$Sale_Price

  expect_error(
    fit <- tabnet_fit(x, y, epochs = 3, lr_scheduler = "step",
                      lr_decay = 0.1, step_size = 1),
    regexp = NA
  )

  sc_fn <- function(optimizer) {
    torch::lr_step(optimizer, step_size = 1, gamma = 0.1)
  }

  expect_error(
    fit <- tabnet_fit(x, y, epochs = 3, lr_scheduler = sc_fn,
                      lr_decay = 0.1, step_size = 1),
    regexp = NA
  )

})


test_that("fit uses config parameters mix from config= and ...", {

  suppressPackageStartupMessages(library(recipes))
  data("attrition", package = "modeldata")
  ids <- sample(nrow(attrition), 256)

  rec <- recipe(EnvironmentSatisfaction ~ ., data = attrition[ids, ]) %>%
    step_normalize(all_numeric(), -all_outcomes())
  fit <- tabnet_fit(rec, attrition, epochs = 1, valid_split = 0.25, verbose = TRUE,
                    config = tabnet_config(decision_width=3, attention_width=5, cat_emb_dim = 2))
  expect_error(
    predict(fit, attrition),
    regexp = NA
  )

  expect_equal(fit$fit$config$verbose, TRUE)
  expect_equal(fit$fit$config$valid_split, 0.25)
  expect_equal(fit$fit$config$n_d, 3)
  expect_equal(fit$fit$config$n_a, 5)

})

test_that("fit works with entmax mask-type", {

  suppressPackageStartupMessages(library(recipes))
  data("attrition", package = "modeldata")
  ids <- sample(nrow(attrition), 256)

  rec <- recipe(EnvironmentSatisfaction ~ ., data = attrition[ids, ]) %>%
    step_normalize(all_numeric(), -all_outcomes())

  expect_error(
    tabnet_fit(rec, attrition, epochs = 1, valid_split = 0.25, verbose = TRUE,
                      config = tabnet_config( mask_type="entmax")),
    regexp = NA
  )
  expect_error(
    predict(tabnet_fit(rec, attrition, epochs = 1, valid_split = 0.25, verbose = TRUE,
                       config = tabnet_config( mask_type="entmax")), attrition),
    regexp = NA
  )

})

test_that("fit raise an error with non-supported mask-type", {

  library(recipes)
  data("attrition", package = "modeldata")
  ids <- sample(nrow(attrition), 256)

  rec <- recipe(EnvironmentSatisfaction ~ ., data = attrition[ids, ]) %>%
    step_normalize(all_numeric(), -all_outcomes())
  expect_error(
    tabnet_fit(rec, attrition, epochs = 1, valid_split = 0.25, verbose = TRUE,
                      config = tabnet_config( mask_type="max_entropy")),
    regexp = "either sparsemax or entmax"
  )

})

test_that("config$loss=`auto` adapt to recipe outcome str()", {

  suppressPackageStartupMessages(library(recipes))
  data("attrition", package = "modeldata")
  ids <- sample(nrow(attrition), 256)

  # nominal outcome
  rec <- recipe(EnvironmentSatisfaction ~ ., data = attrition[ids, ]) %>%
    step_normalize(all_numeric(), -all_outcomes())
  fit_auto <- tabnet_fit(rec, attrition, epochs = 1, verbose = TRUE,
                      config = tabnet_config( loss="auto"))
  expect_equal(fit_auto$fit$config$loss_fn, torch::nn_cross_entropy_loss(), ignore_function_env = TRUE)

  # numerical outcome
  rec <- recipe(MonthlyIncome ~ ., data = attrition[ids, ]) %>%
    step_normalize(all_numeric(), -all_outcomes())
  fit_auto <- tabnet_fit(rec, attrition, epochs = 1, verbose = TRUE,
                      config = tabnet_config( loss="auto"))
  expect_equal(fit_auto$fit$config$loss_fn, torch::nn_mse_loss(), ignore_function_env = TRUE)

})

test_that("config$loss not adapted to recipe outcome raise an explicit error", {

  library(recipes)
  data("attrition", package = "modeldata")
  ids <- sample(nrow(attrition), 256)

  # nominal outcome with numerical loss
  rec <- recipe(EnvironmentSatisfaction ~ ., data = attrition[ids, ]) %>%
    step_normalize(all_numeric(), -all_outcomes())
  expect_error(tabnet_fit(rec, attrition, epochs = 1, verbose = TRUE,
                          config = tabnet_config( loss="mse")),
              regexp = "is not a valid loss for outcome of type"
  )
  # numerical outcome
  rec <- recipe(MonthlyIncome ~ ., data = attrition[ids, ]) %>%
    step_normalize(all_numeric(), -all_outcomes())
  expect_error(tabnet_fit(rec, attrition, epochs = 1, verbose = TRUE,
                      config = tabnet_config( loss="cross_entropy")),
               regexp = "is not a valid loss for outcome of type"
  )
})


test_that("config$loss can be a function", {

  suppressPackageStartupMessages(library(recipes))
  data("attrition", package = "modeldata")
  ids <- sample(nrow(attrition), 256)

  # nominal outcome loss
  rec <- recipe(EnvironmentSatisfaction ~ ., data = attrition[ids, ]) %>%
    step_normalize(all_numeric(), -all_outcomes())
  fit_auto <- tabnet_fit(rec, attrition, epochs = 1, verbose = TRUE,
                      config = tabnet_config( loss=torch::nn_nll_loss()))
  expect_equal(fit_auto$fit$config$loss_fn, torch::nn_nll_loss(), ignore_function_env = TRUE)

  # numerical outcome loss
  rec <- recipe(MonthlyIncome ~ ., data = attrition[ids, ]) %>%
    step_normalize(all_numeric(), -all_outcomes())
  fit_auto <- tabnet_fit(rec, attrition, epochs = 1, verbose = TRUE,
                      config = tabnet_config( loss=torch::nn_poisson_nll_loss()))
  expect_equal(fit_auto$fit$config$loss_fn, torch::nn_poisson_nll_loss(), ignore_function_env = TRUE)

})
