test_that("Check we can use hardhat:::extract_parameter_set_dials() with {dial} tune()ed parameter", {

  model <- tabnet(batch_size = tune(), learn_rate = tune(), epochs = tune(),
                  momentum = tune(), penalty = tune(), rate_step_size = tune()) %>%
    parsnip::set_mode("regression") %>%
    parsnip::set_engine("torch")

  wf <- workflows::workflow() %>%
    workflows::add_model(model) %>%
    workflows::add_formula(Sale_Price ~ .)

  expect_no_error(
    wf %>% hardhat::extract_parameter_set_dials()
  )
})

test_that("Check we can use hardhat:::extract_parameter_set_dials() with {tabnet} tune()ed parameter", {

  model <- tabnet(num_steps = tune(), num_shared = tune(), mask_type = tune(),
                  feature_reusage = tune(), attention_width = tune()) %>%
    parsnip::set_mode("regression") %>%
    parsnip::set_engine("torch")

  wf <- workflows::workflow() %>%
    workflows::add_model(model) %>%
    workflows::add_formula(Sale_Price ~ .)

  expect_no_error(
    wf %>% hardhat::extract_parameter_set_dials()
  )
})

test_that("Check non supported tune()ed parameter raise an explicit error", {

  model <- tabnet(cat_emb_dim = tune(), checkpoint_epochs = 0) %>%
    parsnip::set_mode("regression") %>%
    parsnip::set_engine("torch")

  wf <- workflows::workflow() %>%
    workflows::add_model(model) %>%
    workflows::add_formula(Sale_Price ~ .)

  expect_error(
    wf %>% hardhat::extract_parameter_set_dials(),
    regexp = "cannot be used as a .* parameter yet"
  )
})

