test_that("Training regression", {

  data("ames", package = "modeldata")

  x <- ames[-which(names(ames) == "Sale_Price")]
  y <- ames$Sale_Price

  expect_error(
    fit <- tabnet_fit(x, y, epochs = 1),
    regexp = NA
  )

  expect_error(
    fit <- tabnet_fit(Sale_Price ~ ., data = ames, epochs = 1),
    regexp = NA
  )

  expect_error(
    predict(fit, x),
    regexp = NA
  )

  expect_error(
    fit <- tabnet_fit(x, y, epochs = 2, verbose = TRUE),
    regexp = NA
  )
})

test_that("Training classification", {

  data("attrition", package = "modeldata")

  x <- attrition[-which(names(attrition) == "Attrition")]
  y <- attrition$Attrition

  expect_error(
    fit <- tabnet_fit(x, y, epochs = 1),
    regexp = NA
  )

  expect_error(
    predict(fit, x, type = "prob"),
    regexp = NA
  )

  expect_error(
    predict(fit, x),
    regexp = NA
  )

})

test_that("errors when using an argument that do not exist", {

  data("ames", package = "modeldata")

  x <- ames[-which(names(ames) == "Sale_Price")]
  y <- ames$Sale_Price

  expect_error(
    fit <- tabnet_fit(x, y, epochsas = 1),
    "unused argument"
  )

})

test_that("works with validation split", {

  data("attrition", package = "modeldata")

  x <- attrition[-which(names(attrition) == "Attrition")]
  y <- attrition$Attrition

  expect_error(
    fit <- tabnet_fit(x, y, epochs = 1, valid_split = 0.2),
    regexp = NA
  )

  expect_error(
    fit <- tabnet_fit(x, y, epochs = 1, valid_split = 0.2, verbose = TRUE),
    regexp = NA
  )

})

test_that("can train from a recipe", {

  suppressPackageStartupMessages(library(recipes))
  data("attrition", package = "modeldata")

  rec <- recipe(Attrition ~ ., data = attrition) %>%
    step_normalize(all_numeric(), -all_outcomes())

  expect_error(
    fit <- tabnet_fit(rec, attrition, epochs = 1, valid_split = 0.25,
                    verbose = TRUE),
    regexp = NA
  )

  expect_error(
    predict(fit, attrition),
    regexp = NA
  )

})

test_that("data-frame with missing value makes training fails with explicit message", {

  data("attrition", package = "modeldata")
  ids <- sample(nrow(attrition), 256)

  x <- attrition[ids,-which(names(attrition) == "Attrition")]
  y <- attrition[ids,]$Attrition
  y_missing <- y
  y_missing[1] <- NA

  # numerical missing
  x_missing <- x
  x_missing[1,"Age"] <- NA

  expect_error(
    miss_fit <- tabnet_fit(x_missing, y, epochs = 1),
    regexp = "missing"
  )

  # categorical missing
  x_missing <- x
  x_missing[1,"BusinessTravel"] <- NA

  expect_error(
    miss_fit <- tabnet_fit(x_missing, y, epochs = 1),
    regexp = "missing"
  )

  # missing in outcome
  expect_error(
    miss_fit <- tabnet_fit(x, y_missing, epochs = 1),
    regexp = "missing"
  )

})

test_that("data-frame with missing value makes inference fails with explicit message", {

  data("attrition", package = "modeldata")
  ids <- sample(nrow(attrition), 256)

  x <- attrition[ids,-which(names(attrition) == "Attrition")]
  y <- attrition[ids,]$Attrition
  #
  fit <- tabnet_fit(x, y, epochs = 1)

  # numerical missing
  x_missing <- x
  x_missing[1,"Age"] <- NA

  # predict with numerical missing
  expect_error(
    predict(fit, x_missing),
    regexp = "missing"
  )
  # categorical missing
  x_missing <- x
  x_missing[1,"BusinessTravel"] <- NA

  # predict
  expect_error(
    predict(fit, x_missing),
    regexp = "missing"
  )

})
test_that("inference works with missings in the response vector", {

  suppressPackageStartupMessages(library(recipes))
  data("attrition", package = "modeldata")
  ids <- sample(nrow(attrition), 256)

  rec <- recipe(EnvironmentSatisfaction ~ ., data = attrition[ids, ]) %>%
    step_normalize(all_numeric(), -all_outcomes())
  fit <- tabnet_fit(rec, attrition, epochs = 1, valid_split = 0.25,
                    verbose = TRUE)
  # predict with empty vector
  attrition[["EnvironmentSatisfaction"]] <-NA
  expect_error(
    predict(fit, attrition),
    regexp = NA
  )

  # predict with wrong class
  attrition[["EnvironmentSatisfaction"]] <-NA_character_
  expect_error(
    predict(fit, attrition),
    regexp = NA
  )

  # predict with list column
  attrition[["EnvironmentSatisfaction"]] <- list(NA)
  expect_error(
    predict(fit, attrition),
    regexp = NA
  )

})


test_that("serialization with saveRDS just works", {

  data("ames", package = "modeldata")

  x <- ames[-which(names(ames) == "Sale_Price")]
  y <- ames$Sale_Price

  fit <- tabnet_fit(x, y, epochs = 1)
  predictions <-  predict(fit, ames)

  tmp <- tempfile("model", fileext = "rds")
  saveRDS(fit, tmp)

  rm(fit)
  gc()

  fit2 <- readRDS(tmp)

  expect_equal(
    predictions,
    predict(fit2, ames)
  )

  expect_equal(as.numeric(fit2$fit$network$.check), 1)

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

test_that("checkpoints works for inference", {

  data("ames", package = "modeldata")

  x <- ames[-which(names(ames) == "Sale_Price")]
  y <- ames$Sale_Price

  expect_error(
    fit <- tabnet_fit(x, y, epochs = 3, checkpoint_epochs = 1),
    regexp = NA
  )

  expect_error(
    p1 <- predict(fit, x, epoch = 1),
    regexp = NA
  )

  expect_error(
    p2 <- predict(fit, x, epoch = 2),
    regexp = NA
  )

  expect_error(
    p3 <- predict(fit, x, epoch = 3),
    regexp = NA
  )

  expect_equal(p3, predict(fit, x))

})

test_that("print module works even after a reload from disk", {

  testthat::skip_on_os("linux")
  testthat::skip_on_os("windows")

  data("ames", package = "modeldata")

  x <- ames[-which(names(ames) == "Sale_Price")]
  y <- ames$Sale_Price

  fit <- tabnet_fit(x, y, epochs = 1)

  withr::with_options(new = c(cli.width = 50),
                      expect_snapshot_output(fit))

  tmp <- tempfile("model", fileext = "rds")
  saveRDS(fit, tmp)
  fit2 <- readRDS(tmp)

  withr::with_options(new = c(cli.width = 50),
                      expect_snapshot_output(fit2))


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
