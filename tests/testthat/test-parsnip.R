test_that("multiplication works", {
  library(parsnip)
  data("ames", package = "modeldata")

  expect_error(
    model <- tabnet() %>%
      set_mode("regression") %>%
      set_engine("torch"),
    regexp = NA
  )

  expect_error(
    fit <- model %>%
      fit(Sale_Price ~ ., data = ames),
    regexp = NA
  )

})

test_that("multi_predict works as expected", {

  library(parsnip)

  model <- tabnet() %>%
    set_mode("regression") %>%
    set_engine("torch", checkpoint_epochs = 1)

  data("ames", package = "modeldata")

  expect_error(
    fit <- model %>%
      fit(Sale_Price ~ ., data = ames),
    regexp = NA
  )

  preds <- multi_predict(fit, ames, epochs = c(1,2,3,4,5))

  expect_equal(nrow(preds), nrow(ames))
  expect_equal(nrow(preds$.pred[[1]]), 5)
})

test_that("Check we can finalize a workflow", {

  library(parsnip)
  data("ames", package = "modeldata")

  model <- tabnet(penalty = tune(), epochs = tune()) %>%
    set_mode("regression") %>%
    set_engine("torch")

  wf <- workflows::workflow() %>%
    workflows::add_model(model) %>%
    workflows::add_formula(Sale_Price ~ .)

  wf <- tune::finalize_workflow(wf, tibble::tibble(penalty = 0.01, epochs = 1))

  expect_error(
    fit <- wf %>% fit(data = ames),
    regexp = NA
  )

  expect_equal(rlang::eval_tidy(wf$fit$actions$model$spec$args$penalty), 0.01)
  expect_equal(rlang::eval_tidy(wf$fit$actions$model$spec$args$epochs), 1)
})

test_that("Check we can finalize a workflow from a tune_grid", {

  library(parsnip)
  data("ames", package = "modeldata")

  model <- tabnet(epochs = tune()) %>%
    set_mode("regression") %>%
    set_engine("torch")

  wf <- workflows::workflow() %>%
    workflows::add_model(model) %>%
    workflows::add_formula(Sale_Price ~ .)

  custom_grid <- tidyr::crossing(epochs = c(1,2,3))
  cv_folds <- ames %>%
    rsample::vfold_cv(v = 2, repeats = 1)

  at <- tune::tune_grid(
    object = wf,
    resamples = cv_folds,
    grid = custom_grid,
    metrics = yardstick::metric_set(yardstick::rmse),
    control = tune::control_grid(verbose = F)
  )

  best_rmse <- tune::select_best(at, "rmse")

  expect_error(
    final_wf <- tune::finalize_workflow(wf, best_rmse),
    regexp = NA
  )

  expect_equal(rlang::eval_tidy(final_wf$fit$actions$model$spec$args$epochs), 2)
})
