test_that("get_constr_output handles basic 2D input with identity constraint", {
  x <- torch_tensor(matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2), dtype = torch_float32())
  R <- torch_eye(2, dtype = torch_float32())
  result <- get_constr_output(x, R)
  expect_tensor(result)
  expect_tensor_shape(result, c(2, 2))
  expect_equal_to_r(result, matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2))
})

test_that("get_constr_output applies hierarchy constraint correctly", {
  x <- torch_tensor(matrix(c(1, 5, 3, 2), nrow = 2, ncol = 2, byrow = TRUE), dtype = torch_float64())
  R <- torch_tensor(matrix(c(1, 1, 0, 1), nrow = 2, ncol = 2, byrow = TRUE), dtype = torch_float64())
  result <- get_constr_output(x, R)
  expect_tensor_shape(result, c(2, 2))
  expected <- matrix(c(5, 5, 3, 2), nrow = 2, ncol = 2, byrow = TRUE)
  expect_equal_to_r(result, expected, tolerance = 1e-6)
})

test_that("get_constr_output preserves input dtype", {
  x_f32 <- torch_tensor(matrix(1:4, nrow = 2), dtype = torch_float32())
  x_f64 <- torch_tensor(matrix(1:4, nrow = 2), dtype = torch_float64())
  R <- torch_eye(2)
  expect_tensor_dtype(get_constr_output(x_f32, R), torch_float64())
  expect_tensor_dtype(get_constr_output(x_f64, R), torch_float64())
})

test_that("get_constr_output handles batch dimension correctly", {
  x <- torch_tensor(matrix(1:12, nrow = 3, ncol = 4))
  R <- torch_tensor(matrix(c(1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1), nrow = 4, ncol = 4, byrow = TRUE))
  result <- get_constr_output(x, R)
  expect_tensor_shape(result, c(3, 4))
  
  for (i in 1:3) {
    row_result <- as_array(result[i, ])
    max_grp1 <- max(as_array(x[i, 1:2]))
    max_grp2 <- max(as_array(x[i, 3:4]))
    
    expect_equal(row_result[1:2], rep(max_grp1, 2), tolerance = 1e-6)
    expect_equal(row_result[3:4], rep(max_grp2, 2), tolerance = 1e-6)
  }
})
test_that("get_constr_output works with single sample", {
  x <- torch_tensor(matrix(c(2, 1, 4, 3), nrow = 1, ncol = 4, byrow = TRUE))
  R <- torch_tensor(matrix(c(1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1), nrow = 4, ncol = 4, byrow = TRUE))
  result <- get_constr_output(x, R)
  expect_tensor_shape(result, c(1, 4))
  expected <- matrix(c(2, 2, 4, 4), nrow = 1, byrow = TRUE)
  expect_equal_to_r(result, expected)
})

test_that("get_constr_output handles all-zeros constraint matrix", {
  x <- torch_tensor(matrix(1:6, nrow = 2, ncol = 3))
  R <- torch_zeros(c(3, 3))
  result <- get_constr_output(x, R)
  expect_tensor_shape(result, c(2, 3))
  expect_equal_to_r(result, matrix(0, nrow = 2, ncol = 3))
})

test_that("get_constr_output handles all-ones constraint matrix", {
  x <- torch_tensor(matrix(c(1, 5, 3, 2, 4, 6), nrow = 2, ncol = 3, byrow = TRUE))
  R <- torch_ones(c(3, 3))
  result <- get_constr_output(x, R)
  expect_tensor_shape(result, c(2, 3))
  # Each row is filled with its own row-wise maximum
  expected <- matrix(c(5, 5, 5, 6, 6, 6), nrow = 2, ncol = 3, byrow = TRUE)
  expect_equal_to_r(result, expected, tolerance = 1e-6)
})

test_that("get_constr_output throws error for dimension mismatch", {
  x <- torch_tensor(matrix(1:4, nrow = 2, ncol = 2))
  R <- torch_eye(3)
  expect_error(get_constr_output(x, R), "must match the existing size")
})

test_that("get_constr_output throws error for non-2D R", {
  x <- torch_tensor(matrix(1:4, nrow = 2, ncol = 2))
  R <- torch_tensor(array(1:8, dim = c(2, 2, 2)))
  expect_error(get_constr_output(x, R), "dimension")
})

test_that("max_constraint_output returns original output when ancestor is identity", {
  output <- torch_tensor(matrix(1:6, nrow = 2, ncol = 3))
  labels <- torch_tensor(matrix(c(TRUE, FALSE, TRUE, FALSE, TRUE, FALSE), nrow = 2, ncol = 3), dtype = torch_bool())
  ancestor <- torch_eye(3)
  result <- max_constraint_output(output, labels, ancestor)
  expect_tensor_shape(result, c(2, 3))
  # With an identity ancestor matrix, constraint propagation is neutral.
  # The formula simplifies to: (~labels * output) + (labels * output) == output
  expect_equal_to_r(result, matrix(1:6, nrow = 2, ncol = 3))
})

test_that("max_constraint_output applies constraint to positive labels", {
  output <- torch_tensor(matrix(c(1, 5, 3, 2), nrow = 2, ncol = 2, byrow = TRUE))
  labels <- torch_tensor(matrix(c(1, 0, 1, 0), nrow = 2, ncol = 2, byrow = TRUE), dtype = torch_bool())
  ancestor <- torch_tensor(matrix(c(1, 1, 0, 1), nrow = 2, ncol = 2, byrow = TRUE))
  result <- max_constraint_output(output, labels, ancestor)
  expect_tensor_shape(result, c(2, 2))
  # Unlabelled positions get propagated raw max, labelled get propagated masked max
  expected <- matrix(c(1, 5, 3, 2), nrow = 2, ncol = 2, byrow = TRUE)
  expect_equal_to_r(result, expected)
})

test_that("max_constraint_output handles all-zero labels", {
  output <- torch_tensor(matrix(1:4, nrow = 2, ncol = 2))
  labels <- torch_zeros(c(2, 2), dtype = torch_bool())
  ancestor <- torch_eye(2)
  result <- max_constraint_output(output, labels, ancestor)
  # With all false labels, result equals constr_output. With identity ancestor, constr_output == output
  expect_equal_to_r(result, matrix(1:4, nrow = 2, ncol = 2))
})

test_that("max_constraint_output handles all-one labels", {
  output <- torch_tensor(matrix(c(1, 5, 3, 2), nrow = 2, ncol = 2, byrow = TRUE))
  labels <- torch_ones(c(2, 2), dtype = torch_bool())
  ancestor <- torch_tensor(matrix(c(1, 1, 0, 1), nrow = 2, ncol = 2, byrow = TRUE))
  result <- max_constraint_output(output, labels, ancestor)
  expect_tensor_shape(result, c(2, 2))
  # When all labels are TRUE, (~labels) is 0, so result = train_output.
  expected <- matrix(c(5, 5, 3, 2), nrow = 2, ncol = 2, byrow = TRUE)
  expect_equal_to_r(result, expected)
})

test_that("max_constraint_output preserves output dtype", {
  output_f32 <- torch_tensor(matrix(1:4, nrow = 2), dtype = torch_float32())
  output_f64 <- torch_tensor(matrix(1:4, nrow = 2), dtype = torch_float64())
  labels <- torch_ones(c(2, 2), dtype = torch_bool())
  ancestor <- torch_eye(2)
  expect_tensor_dtype(max_constraint_output(output_f32, labels, ancestor), torch_float64())
  expect_tensor_dtype(max_constraint_output(output_f64, labels, ancestor), torch_float64())
})


test_that("max_constraint_output works with complex hierarchy", {
  output <- torch_tensor(matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3, byrow = TRUE))
  labels <- torch_tensor(matrix(c(1, 0, 0, 0, 1, 0), nrow = 2, ncol = 3, byrow = TRUE), dtype = torch_bool())
  ancestor <- torch_triu(torch_ones(c(3,3)))
  result <- max_constraint_output(output, labels, ancestor)
  expect_tensor_shape(result, c(2, 3))
  # Row 1: label on col 1 -> train_output[1,1]=1, others get constr_output=3
  # Row 2: label on col 2 -> train_output[2,2]=5, others get constr_output=6
  expected <- matrix(c(1, 3, 3,
                       6, 5, 6), nrow = 2, ncol = 3, byrow = TRUE)
  expect_equal_to_r(result, expected)
})

test_that("max_constraint_output handles single element tensors", {
  output <- torch_tensor(matrix(5, nrow = 1, ncol = 1))
  labels <- torch_tensor(matrix(TRUE, nrow = 1, ncol = 1), dtype = torch_bool())
  ancestor <- torch_tensor(matrix(1, nrow = 1, ncol = 1))
  result <- max_constraint_output(output, labels, ancestor)
  expect_tensor_shape(result, c(1, 1))
  # Compare against 1x1 matrix instead of scalar to match torch array output
  expect_equal_to_r(result, matrix(5, nrow = 1, ncol = 1))
})

test_that("max_constraint_output throws error for dimension mismatch", {
  output <- torch_tensor(matrix(1:4, nrow = 2, ncol = 2))
  labels <- torch_ones(c(2, 3), dtype = torch_bool())
  ancestor <- torch_eye(2)
  expect_error(max_constraint_output(output, labels, ancestor), "dimension")
})

test_that("max_constraint_output handles float labels without error", {
  # torch_logical_not works on float tensors (0.0 -> TRUE, others -> FALSE)
  # No explicit type check exists in the function, so it should run successfully
  output <- torch_tensor(matrix(1:4, nrow = 2, ncol = 2))
  labels <- torch_ones(c(2, 2), dtype = torch_float32())
  ancestor <- torch_eye(2)
  expect_silent(max_constraint_output(output, labels, ancestor))
  result <- max_constraint_output(output, labels, ancestor)
  expect_tensor_shape(result, c(2, 2))
})

test_that("get_constr_output and max_constraint_output compose correctly", {
  output <- torch_tensor(matrix(c(1, 4, 2, 3), nrow = 2, ncol = 2, byrow = TRUE))
  labels <- torch_tensor(matrix(c(TRUE, FALSE, TRUE, FALSE), nrow = 2, ncol = 2, byrow = TRUE), dtype = torch_bool())
  ancestor <- torch_tensor(matrix(c(1, 1, 0, 1), nrow = 2, ncol = 2, byrow = TRUE))
  direct <- max_constraint_output(output, labels, ancestor)
  constr_out <- get_constr_output(output, ancestor)
  train_out <- get_constr_output(labels * output, ancestor)
  manual <- torch_logical_not(labels) * constr_out + labels * train_out
  expect_equal_to_r(direct, as_array(manual))
})

test_that("get_constr_output handles negative values correctly", {
  x <- torch_tensor(matrix(c(-5, -1, -3, -2), nrow = 2, ncol = 2, byrow = TRUE))
  R <- torch_tensor(matrix(c(1, 1, 0, 1), nrow = 2, ncol = 2, byrow = TRUE))
  result <- get_constr_output(x, R)
  expected <- matrix(c(-1, 0, -2, 0), nrow = 2, ncol = 2, byrow = TRUE)
  expect_equal_to_r(result, expected)
})

test_that("max_constraint_output handles mixed positive-negative with constraints", {
  output <- torch_tensor(matrix(c(-5, -3, -1, 4), nrow = 2, ncol = 2, byrow = TRUE))
  labels <- torch_tensor(matrix(c(TRUE, TRUE, FALSE, TRUE), nrow = 2, ncol = 2, byrow = TRUE), dtype = torch_bool())
  ancestor <- torch_tensor(matrix(c(1, 1, 0, 1), nrow = 2, ncol = 2, byrow = TRUE))
  result <- max_constraint_output(output, labels, ancestor)
  expected <- matrix(c(-3, 0, 4, 4), nrow = 2, ncol = 2, byrow = TRUE)
  expect_equal_to_r(result, expected)
})

# need rework as FromDataFrameNetwork(edges) gives "cannot find root name" error
test_that("build-ancestor-matrix diagonal is always 1 for every class", {
  edges <- data.frame(from = c(1L, 2L, 2L), 
                      to   = c(2L, 3L, 4L))
  R <- build_ancestor_matrix(FromDataFrameNetwork(mutate_all(edges, as.character)))
  R_dense <- R$to_dense()

  expect_equal_to_r(R_dense[1, 1], TRUE)
  expect_equal_to_r(R_dense[2, 2], TRUE)
  expect_equal_to_r(R_dense[3, 3], TRUE)
})

test_that("build-ancestor-matrix: single edge produces correct transitive pair", {
  # 1 -> 2 means "2 is ancestor of 1", so transposed: R[2, 1] = 1
  edges <- data.frame(from = c(1L, 2L), to = c(2L, 3L))
  R <- build_ancestor_matrix(FromDataFrameNetwork(mutate_all(edges, as.character)))
  R_dense <- R$to_dense()

  # 2 is descendant of 2 (self)
  expect_equal_to_r(R_dense[2, 2], TRUE)
  # 1 is descendant of 1 (self)
  expect_equal_to_r(R_dense[1, 1], TRUE)
  # 1 is descendant of 2 (because 1 -> 2)
  expect_equal_to_r(R_dense[2, 1], TRUE)
  # 2 is NOT descendant of 1
  expect_equal_to_r(R_dense[1, 2], FALSE)
})

test_that("build-ancestor-matrix: multi-hop ancestor chain is fully resolved", {
  # Chain: 2 -> 3 -> 4 -> 5 (each is ancestor of the previous)
  # After transpose: 4 is descendant of 1, 2, 3, 4
  #                  3 is descendant of 1, 2, 3
  #                  2 is descendant of 1, 2
  #                  1 is descendant of 1
  edges <- data.frame(
    from = c(1L, 2L, 3L, 4L),
    to   = c(2L, 3L, 4L, 5L)
  )
  R <- build_ancestor_matrix(FromDataFrameNetwork(mutate_all(edges, as.character)))
  R_dense <- R$to_dense()

  # Row 1: only node 1 is its own descendant
  expect_equal_to_r(R_dense, lower.tri(diag(4), diag = TRUE))
})

test_that("build-ancestor-matrix: diamond hierarchy merges both paths", {
  # Diamond: 1 -> 2 -> 4, 1 -> 3 -> 4
  # After transpose: 4 is descendant of all; 2 and 3 are descendants of
  #   1 and themselves only
  edges <- data.frame(
    from = c(1L, 1L, 2L, 3L),
    to   = c(2L, 3L, 4L, 4L)
  )
  R <- build_ancestor_matrix(FromDataFrameNetwork(mutate_all(edges, as.character)))
  R_dense <- R$to_dense()

  expect_equal_to_r(R_dense[3, 1], TRUE)
  expect_equal_to_r(R_dense[4, 2], TRUE)
  expect_equal_to_r(R_dense[4, 3], FALSE)
  expect_equal_to_r(R_dense[2, 4], FALSE)
})

# test_that("build-ancestor-matrix: isolated nodes have only a diagonal entry", {
#   edges <- data.frame(from = c(1L, 1L),
#                       to   = c(2L, 1L))
#   R <- build_ancestor_matrix(FromDataFrameNetwork(mutate_all(edges, as.character)))
#   R_dense <- R$to_dense()
# 
#   # Nodes 3, 4, 5 have no edges
#   expect_equal_to_r(R_dense[3, 3], TRUE)
#   expect_equal_to_r(R_dense[3, ], c(FALSE, FALSE, TRUE, FALSE, FALSE))
#   expect_equal_to_r(R_dense[4, 4], TRUE)
#   expect_equal_to_r(R_dense[5, 5], TRUE)
# })
# 
# test_that("build-ancestor-matrix: n_classes defaults to max node id when NULL", {
#   edges <- data.frame(from = c(1L, 1L), to = c(5L, 1L))
#   R <- build_ancestor_matrix(FromDataFrameNetwork(mutate_all(edges, as.character)))
# 
#   # n_classes should be max(1, 5, TRUE) = 5
#   expect_tensor_shape(R, c(5, 5))
# })

# test_that("build-ancestor-matrix: output has correct shape and dtype", {
#   edges <- data.frame(from = c(1L, 2L), to = c(2L, 1L))
#   R <- build_ancestor_matrix(FromDataFrameNetwork(mutate_all(edges, as.character)))
#   
#   expect_tensor_shape(R, c(3L, 3L))
#   expect_tensor_dtype(R, torch::torch_bool())
#   expect_true(R$is_sparse())
# })
# 
test_that("build-ancestor-matrix: output uses 0-based indices internally", {
  # Verify that torch sees correct values when converted to dense
  edges <- data.frame(from = c(1L, 2L, 1L), to = c(2L, 3L, 3L))
  R <- build_ancestor_matrix(FromDataFrameNetwork(mutate_all(edges, as.character)))
  expect_equal_to_r(R$to_dense(), matrix(c(TRUE, TRUE, FALSE, TRUE), nrow=2))
})

# test_that("build-ancestor-matrix: single-node graph produces identity-like matrix", {
#   edges <- data.frame(from = 1L, to = 1L)
#   R <- build_ancestor_matrix(FromDataFrameNetwork(mutate_all(edges, as.character)))
#   expect_tensor_shape(R$to_dense(), c(1L, 1L))
#   expect_equal_to_r(R$to_dense()[1, 1], TRUE)
# })

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

