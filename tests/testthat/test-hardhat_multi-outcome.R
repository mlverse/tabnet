
test_that("Training multi-output regression from data.frame", {

  expect_no_error(
    fit <- tabnet_fit(x, data.frame(y = y, z = y + 1), epochs = 1)
  )

  expect_no_error(
    result <- predict(fit, x)
  )
  expect_equal(ncol(result), 2)

})

test_that("Training multi-output regression from formula", {

  expect_no_error(
    fit <- tabnet_fit(Sale_Price + Latitude + Longitude ~ ., small_ames, epochs = 1)
  )

  expect_no_error(
    result <- predict(fit, ames)
  )
  expect_equal(ncol(result), 3)

})

test_that("Training multi-output regression from recipe", {

  rec <- recipe(Sale_Price + Latitude + Longitude ~ ., data = small_ames) %>%
    step_zv(all_predictors()) %>%
    step_normalize(all_numeric(), -all_outcomes())

  expect_no_error(
    fit <- tabnet_fit(rec, small_ames, epochs = 1)
  )

  expect_no_error(
    result <- predict(fit, ames)
  )
  expect_equal(ncol(result), 3)

})

test_that("Training multilabel classification from data.frame", {

  expect_no_error(
    fit <- tabnet_fit(attri_mult_x, data.frame(y = attriy, z = attriy, sat = attrix$JobSatisfaction),
                      epochs = 1)
  )

  expect_no_error(
    result <- predict(fit, attri_mult_x, type = "prob")
  )

  expect_equal(ncol(result), 3)
  outcome_nlevels <- purrr::map_dbl(fit$blueprint$ptypes$outcomes, ~length(levels(.x)))
  # we get back outcomes vars with a `.pred_` prefix
  expect_equal(stringr::str_remove(names(result), ".pred_"), names(outcome_nlevels))

  # result columns are tibbles of resp 2, 2, 4 columns
  expect_true(all(purrr::map_lgl(result, tibble::is_tibble)))
  expect_equal(purrr::map_dbl(result, ncol), outcome_nlevels, ignore_attr = TRUE)

  expect_no_error(
    result <- predict(fit, attri_mult_x)
  )
  expect_equal(ncol(result), 3)
  expect_equal(stringr::str_remove(names(result), ".pred_class_"), names(outcome_nlevels))

})

test_that("Training multilabel classification from formula", {

  expect_no_error(
    fit <- tabnet_fit(Attrition + JobSatisfaction ~ ., data = attrition[ids,],
                      epochs = 1)
  )

  expect_no_error(
    result <- predict(fit, attri_mult_x, type = "prob")
  )

  expect_equal(ncol(result), 2)
  outcome_nlevels <- purrr::map_dbl(fit$blueprint$ptypes$outcomes, ~length(levels(.x)))
  # we get back outcomes vars with a `.pred_` prefix
  expect_equal(stringr::str_remove(names(result), ".pred_"), names(outcome_nlevels))

  # result columns are tibbles of resp 2, 2, 4 columns
  expect_true(all(purrr::map_lgl(result, tibble::is_tibble)))
  expect_equal(purrr::map_dbl(result, ncol), outcome_nlevels, ignore_attr = TRUE)

})

test_that("Training multilabel classification from recipe", {

  rec <- recipe(Attrition + JobSatisfaction ~ ., data = attrition[ids,]) %>%
    step_zv(all_predictors()) %>%
    step_normalize(all_numeric(), -all_outcomes())

  expect_no_error(
    fit <- tabnet_fit(rec, attrition[ids,], epochs = 1, valid_split = 0.25,
                      verbose = TRUE)
  )

  expect_no_error(
    result <- predict(fit, attrition)
  )
  expect_equal(ncol(result), 2)

  outcome_nlevels <- purrr::map_dbl(fit$blueprint$ptypes$outcomes, ~length(levels(.x)))
  expect_equal(stringr::str_remove(names(result), ".pred_class_"), names(outcome_nlevels))

})

test_that("Training multilabel classification from data.frame with validation split", {

  expect_no_error(
    fit <- tabnet_fit(attri_mult_x, data.frame(y=attriy, z=attriy, sat=attrix$JobSatisfaction),
                      valid_split = 0.2, epochs = 1)
  )

  expect_no_error(
    result <- predict(fit, attri_mult_x, type = "prob")
  )

  expect_equal(ncol(result), 3)

  outcome_nlevels <- purrr::map_dbl(fit$blueprint$ptypes$outcomes, ~length(levels(.x)))
  # we get back outcomes vars with a `.pred_` prefix
  expect_equal(stringr::str_remove(names(result), ".pred_"), names(outcome_nlevels))

  # result columns are tibbles of resp 2, 2, 4 columns
  expect_true(all(purrr::map_lgl(result, tibble::is_tibble)))
  expect_equal(purrr::map_dbl(result, ncol), outcome_nlevels, ignore_attr = TRUE)

  expect_no_error(
    result <- predict(fit, attri_mult_x)
  )
  expect_equal(ncol(result), 3)

  # we get back outcomes vars with a `.pred_class_` prefix
  expect_equal(stringr::str_remove(names(result), ".pred_class_"), names(fit$blueprint$ptypes$outcomes))
})


test_that("Training multilabel mixed output fails with explicit error", {

  attri_multi_x <- attrix[-which(names(attrix) == "PercentSalaryHike")]
  expect_error(
    fit <- tabnet_fit(attri_multi_x, data.frame(y = attriy, hik = attrix$PercentSalaryHike), epochs = 1),
    "Mixed multi-outcome type"
  )
  expect_error(
    fit <- tabnet_fit(Attrition + PercentSalaryHike ~ ., data = attrition[ids,], epochs = 1),
    "Mixed multi-outcome type"
  )
  rec <- recipe(Attrition + PercentSalaryHike ~ ., data = attrition[ids,]) %>%
    step_normalize(all_numeric(), -all_outcomes())
  expect_error(
    fit <- tabnet_fit(rec, data = attrition[ids,], epochs = 1),
    "Mixed multi-outcome type"
  )
})

test_that("Training multi-output regression fails for matrix", {

  expect_error(
    fit <- tabnet_fit(x, matrix(rnorm( 2 * length(y)), ncol = 2), epochs = 1),
    "All columns of `y` must have unique names"
  )

  expect_error(
    fit <- tabnet_fit(x, matrix(factor(runif( 2 * length(y)) < 0.5) , ncol = 2), epochs = 1),
    "All columns of `y` must have unique names"
  )

})
