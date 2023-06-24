test_that("Training hierarchical classification for {data.tree} Node", {

  expect_no_error(
    fit <- tabnet_fit(acme, epochs = 1)
  )

  expect_no_error(
    result <- predict(fit, acme, type = "prob")
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

test_that("Training hierarchical classification for {data.tree} Node with validation split", {

  expect_no_error(
    fit <- tabnet_fit(acme, data.frame(y=attriy, z=attriy, sat=attrix$JobSatisfaction),
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

test_that("Training hierarchical classification works with wierd colnames", {

  # augment acme dataset with a forbidden column name
  acme$Do(function(x) {
    x$level_4 <- data.tree::Aggregate(node = x,
                           attribute = "p",
                           aggFun = sum)
  },
  traversal = "post-order")
  expect_error(
    tabnet_fit(acme, valid_split = 0.2, epochs = 1)
    ,"Error")

})

