test_that("Training hierarchical classification for {data.tree} Node attrition_tree", {
  expect_no_error(
    fit <- tabnet_fit(attrition_tree, epochs = 1)
  )
  expect_no_error(
    result <- predict(fit, attrition_tree, type = "prob")
  )

  expect_equal(ncol(result), 2) # 2 outcomes levels_

  outcome_nlevels <- purrr::map_dbl(fit$blueprint$ptypes$outcomes, ~nlevels(.x))
  # we get back outcomes vars with a `.pred_` prefix
  expect_equal(stringr::str_remove(names(result), ".pred_"), names(outcome_nlevels))

  # result columns are tibbles of resp 2, 2, 4 columns
  expect_true(all(purrr::map_lgl(result, tibble::is_tibble)))
  expect_equal(unname(purrr::map_dbl(result, ncol)), unname(outcome_nlevels), ignore_attr = TRUE)

})

test_that("Training hierarchical classification for {data.tree} Node with validation split", {

  expect_no_error(
    fit <- tabnet_fit(attrition_tree, valid_split = 0.2, epochs = 1)
  )
  expect_named(fit$fit$config, "ancestor")
  expect_true(fit$fit$config$ancestor$is_sparse())
  
  expect_no_error(
    result <- predict(fit, attrition_tree, type = "prob")
  )

  expect_equal(ncol(result), 2) # 2 outcomes levels_

  outcome_nlevels <- purrr::map_dbl(fit$blueprint$ptypes$outcomes, ~nlevels(.x))
  # we get back outcomes vars with a `.pred_` prefix
  expect_equal(stringr::str_remove(names(result), ".pred_"), names(outcome_nlevels))

  # result columns are tibbles of resp 2, 2, 4 columns
  expect_true(all(purrr::map_lgl(result, tibble::is_tibble)))
  expect_equal(unname(purrr::map_dbl(result, ncol)), unname(outcome_nlevels), ignore_attr = TRUE)

  expect_no_error(
    result <- predict(fit, attrition_tree)
  )
  expect_equal(ncol(result), 2) # 2 outcomes levels_

  # we get back outcomes vars with a `.pred_class_` prefix
  expect_equal(stringr::str_remove(names(result), ".pred_class_"), names(fit$blueprint$ptypes$outcomes))
})

test_that("hierarchical classification for {data.tree} Node is explainable", {

    fit <- tabnet_fit(attrition_tree, epochs = 1)

  expect_no_error(
    explain <- tabnet_explain(fit, attrition_tree)
  )

  expect_no_error(
    autoplot(explain)
  )

})

test_that("we properly check non-compliant colnames", {

  # try to use starwars dataset with two forbidden column name
  starwars_tree <- starwars %>%
    mutate(pathString = paste("tree", species, homeworld, `name`, sep = "/"))
  expect_error(
    check_compliant_node(starwars_tree)
    ,"reserved names")

  # augment acme dataset with a forbidden column name with no impact on predictor is ok
  acme$Do(function(x) {
    x$level_4 <- as.character(data.tree::Aggregate(node = x,
                           attribute = "p",
                           aggFun = sum))
  },
  traversal = "post-order")
  expect_no_error(check_compliant_node(acme))

  expect_no_error(tabnet_fit(acme, epochs = 1))

  # augment acme dataset with a used forbidden column name raise error
  acme$Do(function(x) {
    x$level_3 <- data.tree::Aggregate(node = x,
                           attribute = "p",
                           aggFun = sum)
  },
  traversal = "post-order")
  expect_error(
    check_compliant_node(acme)
    ,"reserved names")

  expect_error(
    tabnet_fit(acme, epochs = 1)
    ,"reserved names")

})

