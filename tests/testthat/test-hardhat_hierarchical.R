test_that("C-HMCNN get_constr_output works ", {
  x <- torch::torch_rand(c(2,4))
  R <- torch::torch_tril(torch::torch_zeros(c(4,4))$bernoulli(p = 0.2) + torch::torch_diag(rep(1,4)))$to(dtype = torch::torch_bool())
  expect_no_error(
    constr_output <- get_constr_output(x, R)
  )
  expect_tensor_shape(
    constr_output, x$shape
  )
  # expect_equal(
  #   constr_output$dtype, torch_tensor(0.1)$dtype
  # )

  R <- torch::torch_zeros(c(4,4))$to(dtype = torch::torch_bool())
  expect_equal_to_tensor(
    get_constr_output(x, R), torch::torch_zeros_like(x)
  )
})

test_that("C-HMCNN max_constraint_output works ", {
  output <- torch::torch_rand(c(3, 5))
  labels <- torch::torch_diag(rep(1,5))[1:3, ]$to(dtype = torch::torch_bool())
  ancestor <- torch::torch_tril(torch::torch_zeros(c(5, 5))$bernoulli(p = 0.2) )$to(dtype = torch::torch_bool())

  expect_no_error(
    MC_output <- max_constraint_output(output, labels, ancestor)
  )
  expect_tensor_shape(
    MC_output, output$shape
  )
  # max_constraint_output is not identity
  expect_not_equal_to_tensor(
    MC_output, output
  )
  # max_constraint_output provides more than 35% null values
  expect_gte(
    as.matrix(torch::torch_sum(MC_output == 0), device="cpu"), .30 * output$shape[1] * output$shape[2]
  )
})

test_that("node_to_df works ", {
  expect_no_error(
    node_to_df(acme)
  )
  expect_no_error(
    attrition_df <- node_to_df(attrition_tree)
  )
  # node_to_df removes first and last level of the hierarchy
  outcome_levels <- paste0("level_", seq(2, attrition_tree$height - 1))
  expect_equal(names(attrition_df$y), outcome_levels)

  # node_to_df do not shuffle outcome rows
  df <- tibble(pred_1 = seq(1,26), pred_2 = seq(26,1),
               level_2 = factor(LETTERS[1:26]), level_3 = factor(letters[26:1]))
  df_node_df <- df %>%
    mutate(pathString = paste("synth", level_2, level_3, level_3, sep = "/")) %>%
    select(-level_2, -level_3) %>%
    as.Node() %>%
    node_to_df()

  expect_equal(df_node_df$y %>% as_tibble(), df %>% select(starts_with("level_")))
  expect_equal(df_node_df$x %>% as_tibble(), df %>% select(starts_with("pred_")))

})


test_that("Training hierarchical classification for {data.tree} Node", {

  expect_no_error(
    fit <- tabnet_fit(acme, epochs = 1)
  )
  expect_no_error(
    result <- predict(fit, acme_df, type = "prob")
  )

  expect_equal(ncol(result), 3)
  outcome_levels <-levels(fit$blueprint$ptypes$outcomes[[1]])
  # we get back outcomes vars with a `.pred_` prefix
  expect_equal(stringr::str_remove(names(result), ".pred_"), outcome_levels)
  expect_no_error(
    result <- predict(fit, acme_df)
  )
  expect_equal(ncol(result), 1)

  expect_no_error(
    fit <- tabnet_fit(attrition_tree, epochs = 1)
  )
  expect_no_error(
    result <- predict(fit, attrition_tree, type = "prob")
  )

  expect_equal(ncol(result), 2) # 2 outcomes levels_

  outcome_nlevels <- purrr::map_dbl(fit$blueprint$ptypes$outcomes, ~length(levels(.x)))
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

  expect_no_error(
    result <- predict(fit, attrition_tree, type = "prob")
  )

  expect_equal(ncol(result), 2) # 2 outcomes levels_

  outcome_nlevels <- purrr::map_dbl(fit$blueprint$ptypes$outcomes, ~length(levels(.x)))
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

