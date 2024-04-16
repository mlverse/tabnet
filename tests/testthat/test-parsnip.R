test_that("parsnip fit model works", {

  # default params
  expect_no_error(
    model <- tabnet() %>%
      parsnip::set_mode("regression") %>%
      parsnip::set_engine("torch")
  )

  expect_no_error(
    fit <- model %>%
      parsnip::fit(Sale_Price ~ ., data = small_ames)
  )

  # some setup params
  expect_no_error(
    model <- tabnet(epochs = 2, learn_rate = 1e-5) %>%
      parsnip::set_mode("regression") %>%
      parsnip::set_engine("torch")
  )

  expect_no_error(
    fit <- model %>%
      parsnip::fit(Sale_Price ~ ., data = small_ames)
  )

  # new batch of setup params
  expect_no_error(
    model <- tabnet(penalty = 0.2, verbose = FALSE, early_stopping_tolerance = 1e-3) %>%
      parsnip::set_mode("classification") %>%
      parsnip::set_engine("torch")
  )

  expect_no_error(
    fit <- model %>%
      parsnip::fit(Overall_Cond ~ ., data = small_ames)
  )

})

test_that("parsnip fit model works from a pretrained model", {

  # default params
  expect_no_error(
    model <- tabnet(tabnet_model = ames_pretrain, from_epoch = 1, epoch = 1) %>%
      parsnip::set_mode("regression") %>%
      parsnip::set_engine("torch")
  )

  expect_no_error(
    fit <- model %>%
      parsnip::fit(Sale_Price ~ ., data = small_ames)
  )




})

test_that("multi_predict works as expected", {

  model <- tabnet(checkpoint_epoch = 1) %>%
    parsnip::set_mode("regression") %>%
    parsnip::set_engine("torch")

  expect_no_error(
    fit <- model %>%
      parsnip::fit(Sale_Price ~ ., data = small_ames)
  )

  preds <- parsnip::multi_predict(fit, small_ames, epochs = c(1,2,3,4,5))

  expect_equal(nrow(preds), nrow(small_ames))
  expect_equal(nrow(preds$.pred[[1]]), 5)
})

test_that("Check we can finalize a workflow", {

  model <- tabnet(penalty = tune(), epochs = tune()) %>%
    parsnip::set_mode("regression") %>%
    parsnip::set_engine("torch")

  wf <- workflows::workflow() %>%
    workflows::add_model(model) %>%
    workflows::add_formula(Sale_Price ~ .)

  wf <- tune::finalize_workflow(wf, tibble::tibble(penalty = 0.01, epochs = 1))

  expect_no_error(
    fit <- wf %>% parsnip::fit(data = small_ames)
  )

  expect_equal(rlang::eval_tidy(wf$fit$actions$model$spec$args$penalty), 0.01)
  expect_equal(rlang::eval_tidy(wf$fit$actions$model$spec$args$epochs), 1)
})

test_that("Check we can finalize a workflow from a tune_grid", {

  model <- tabnet(epochs = tune(), checkpoint_epochs = 1) %>%
    parsnip::set_mode("regression") %>%
    parsnip::set_engine("torch")

  wf <- workflows::workflow() %>%
    workflows::add_model(model) %>%
    workflows::add_formula(Sale_Price ~ .)

  custom_grid <- tidyr::crossing(epochs = c(1,2,3))
  cv_folds <- small_ames %>%
    rsample::vfold_cv(v = 2, repeats = 1)

  at <- tune::tune_grid(
    object = wf,
    resamples = cv_folds,
    grid = custom_grid,
    metrics = yardstick::metric_set(yardstick::rmse),
    control = tune::control_grid(verbose = F)
  )

  best_rmse <- tune::select_best(at, metric = "rmse")

  expect_no_error(
    final_wf <- tune::finalize_workflow(wf, best_rmse)
  )
})

test_that("tabnet grid reduction - torch", {

  mod <- tabnet() %>%
    parsnip::set_engine("torch")

  # A typical grid
  reg_grid <- expand.grid(epochs = 1:3, penalty = 1:2)
  reg_grid_smol <- tune::min_grid(mod, reg_grid)

  expect_equal(reg_grid_smol$epochs, rep(3, 2))
  expect_equal(reg_grid_smol$penalty, 1:2)
  for (i in 1:nrow(reg_grid_smol)) {
    expect_equal(reg_grid_smol$.submodels[[i]], list(epochs = 1:2))
  }

  # Unbalanced grid
  reg_ish_grid <- expand.grid(epochs = 1:3, penalty = 1:2)[-3, ]
  reg_ish_grid_smol <- tune::min_grid(mod, reg_ish_grid)

  expect_equal(reg_ish_grid_smol$epochs, 2:3)
  expect_equal(reg_ish_grid_smol$penalty, 1:2)
  for (i in 2:nrow(reg_ish_grid_smol)) {
    expect_equal(reg_ish_grid_smol$.submodels[[i]], list(epochs = 1:2))
  }

  # Grid with a third parameter
  reg_grid_extra <- expand.grid(epochs = 1:3, penalty = 1:2, batch_size = 10:12)
  reg_grid_extra_smol <- tune::min_grid(mod, reg_grid_extra)

  expect_equal(reg_grid_extra_smol$epochs, rep(3, 6))
  expect_equal(reg_grid_extra_smol$penalty, rep(1:2, each = 3))
  expect_equal(reg_grid_extra_smol$batch_size, rep(10:12, 2))
  for (i in 1:nrow(reg_grid_extra_smol)) {
    expect_equal(reg_grid_extra_smol$.submodels[[i]], list(epochs = 1:2))
  }

  # Only epochs
  only_epochs <- expand.grid(epochs = 1:3)
  only_epochs_smol <- tune::min_grid(mod, only_epochs)

  expect_equal(only_epochs_smol$epochs, 3)
  expect_equal(only_epochs_smol$.submodels, list(list(epochs = 1:2)))

  # No submodels
  no_sub <- tibble::tibble(epochs = 1, penalty = 1:2)
  no_sub_smol <- tune::min_grid(mod, no_sub)

  expect_equal(no_sub_smol$epochs, rep(1, 2))
  expect_equal(no_sub_smol$penalty, 1:2)
  for (i in 1:nrow(no_sub_smol)) {
    expect_length(no_sub_smol$.submodels[[i]], 0)
  }

  # different id names
  mod_1 <- tabnet(epochs = tune("Amos")) %>%
    parsnip::set_engine("torch")
  reg_grid <- expand.grid(Amos = 1:3, penalty = 1:2)
  reg_grid_smol <- tune::min_grid(mod_1, reg_grid)

  expect_equal(reg_grid_smol$Amos, rep(3, 2))
  expect_equal(reg_grid_smol$penalty, 1:2)
  for (i in 1:nrow(reg_grid_smol)) {
    expect_equal(reg_grid_smol$.submodels[[i]], list(Amos = 1:2))
  }

  all_sub <- expand.grid(Amos = 1:3)
  all_sub_smol <- tune::min_grid(mod_1, all_sub)

  expect_equal(all_sub_smol$Amos, 3)
  expect_equal(all_sub_smol$.submodels[[1]], list(Amos = 1:2))

  mod_2 <- tabnet(epochs = tune("Ade Tukunbo")) %>%
    parsnip::set_engine("torch")
  reg_grid <- expand.grid(`Ade Tukunbo` = 1:3, penalty = 1:2, ` \t123` = 10:11)
  reg_grid_smol <- tune::min_grid(mod_2, reg_grid)

  expect_equal(reg_grid_smol$`Ade Tukunbo`, rep(3, 4))
  expect_equal(reg_grid_smol$penalty, rep(1:2, each = 2))
  expect_equal(reg_grid_smol$` \t123`, rep(10:11, 2))
  for (i in 1:nrow(reg_grid_smol)) {
    expect_equal(reg_grid_smol$.submodels[[i]], list(`Ade Tukunbo` = 1:2))
  }
})

test_that("Check workflow can use case_weight", {

  small_ames_cw <- small_ames %>% dplyr::mutate(case_weight = hardhat::frequency_weights(Year_Sold))
  model <- tabnet(epochs = 2) %>%
    parsnip::set_mode("regression") %>%
    parsnip::set_engine("torch")

  wf <- workflows::workflow() %>%
    workflows::add_model(model) %>%
    workflows::add_formula(Sale_Price ~ .) %>%
    workflows::add_case_weights(case_weight)

  expect_no_error(
    fit <- wf %>% parsnip::fit(data = small_ames_cw)
  )
  expect_no_error(
    predict(fit, small_ames)
  )


})
