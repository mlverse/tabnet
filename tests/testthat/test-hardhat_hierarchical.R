test_that("Training hierarchical classification for {data.tree} Node", {

  expect_no_error(
    fit <- tabnet_fit(acme, epochs = 1)
  )
  expect_no_error(
    result <- predict(fit, acme_df, type = "prob")
  )

  expect_equal(ncol(result), 3)
  outcome_nlevels <- purrr::map_dbl(fit$blueprint$ptypes$outcomes, ~length(levels(.x)))
  # we get back outcomes vars with a `.pred_` prefix
  expect_equal(stringr::str_remove(names(result), ".pred_"), names(outcome_nlevels))

  # result columns are tibbles of resp 2, 2, 4 columns
  expect_true(all(purrr::map_lgl(result, tibble::is_tibble)))
  expect_equal(purrr::map_dbl(result, ncol), outcome_nlevels, ignore_attr = TRUE)

  expect_no_error(
    result <- predict(fit, acme_df)
  )
  expect_equal(ncol(result), 3)
  expect_equal(stringr::str_remove(names(result), ".pred_class_"), names(outcome_nlevels))

})

test_that("Training hierarchical classification for {data.tree} Node with validation split", {

  expect_no_error(
    fit <- tabnet_fit(acme, valid_split = 0.2, epochs = 1)
  )

  expect_no_error(
    result <- predict(fit, acme_df, type = "prob")
  )

  expect_equal(ncol(result), 3)

  outcome_nlevels <- purrr::map_dbl(fit$blueprint$ptypes$outcomes, ~length(levels(.x)))
  # we get back outcomes vars with a `.pred_` prefix
  expect_equal(stringr::str_remove(names(result), ".pred_"), names(outcome_nlevels))

  # result columns are tibbles of resp 2, 2, 4 columns
  expect_true(all(purrr::map_lgl(result, tibble::is_tibble)))
  expect_equal(purrr::map_dbl(result, ncol), outcome_nlevels, ignore_attr = TRUE)

  expect_no_error(
    result <- predict(fit, acme_df)
  )
  expect_equal(ncol(result), 3)

  # we get back outcomes vars with a `.pred_class_` prefix
  expect_equal(stringr::str_remove(names(result), ".pred_class_"), names(fit$blueprint$ptypes$outcomes))
})

test_that("we can check with non-compliant colnames", {

  # try to use starwars dataset with two forbidden column name
  starwars_tree <- starwars %>%
    mutate(pathString = paste("tree", species, homeworld, `name`, sep = "/"))
  expect_error(
    check_compliant_node(starwars_tree)
    ,"reserved names")

  # augment acme dataset with a forbidden column name
  acme$Do(function(x) {
    x$level_4 <- data.tree::Aggregate(node = x,
                           attribute = "p",
                           aggFun = sum)
  },
  traversal = "post-order")
  expect_error(
    check_compliant_node(acme)
    ,"reserved names")

  expect_error(
    tabnet_fit(acme, valid_split = 0.2, epochs = 1)
    ,"reserved names")

})

