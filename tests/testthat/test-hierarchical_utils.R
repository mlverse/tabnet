test_that("build_ancestor_matrix_from_outcomes handles basic unrelated hierarchy as a diag matrix", {
  tree_df <- data.frame(pathString = c("Root/A/C1", "Root/A/C2","Root/B/D1", "Root/B/D2"))
  tree <- as.Node(tree_df)
  
  result <- build_ancestor_matrix_from_outcomes(tree)

  expect_tensor_shape(result, c(1,2,2))
  expect_equal_to_r(result$squeeze(), diag(2))
})

test_that("build_ancestor_matrix_from_outcomes handles linear chain of internal nodes", {

  tree_df <- data.frame(pathString = c("Root/A/B", "Root/A/B/C"), value = 1:2)
  tree <- as.Node(tree_df)
  
  result <- build_ancestor_matrix_from_outcomes(tree)

  # lower triangular 2 x 2 mat
  expected <- fBasics::triang(matrix(1, nrow = 2, ncol = 2))
  expect_equal_to_r(result$squeeze(), expected)
})

test_that("build_ancestor_matrix_from_outcomes calculates transitive closure correctly", {

  tree_df <- data.frame(pathString = c("Root/A/B/C", "Root/A/B/C/D"), value = 1:2)
  tree <- as.Node(tree_df)
  
  result <- build_ancestor_matrix_from_outcomes(tree)
  
  # lower triangular 3 x 3 mat
  expected <- fBasics::triang(matrix(TRUE, nrow = 3, ncol = 3))
  expect_equal_to_r(result$squeeze(), expected)
})

test_that("build_ancestor_matrix_from_outcomes handles branching internal nodes", {
  
  tree_df <- data.frame(pathString = c("Root/A/C/E1", "Root/A/C/E2", "Root/B/D/E1", "Root/B/D/E3"))
  tree <- as.Node(tree_df)
  
  result <- build_ancestor_matrix_from_outcomes(tree)
  
  # diagonal matrix with 2 ancestors
  expected <- diag(4)
  expected[2,1] <- 1L
  expected[4,3] <- 1L

  expect_equal_to_r(result$squeeze(), expected)
})

test_that("build_ancestor_matrix_from_outcomes returns empty for Root-only tree", {
  tree <- Node$new("Root")
  result <- build_ancestor_matrix_from_outcomes(tree)
  expect_equal(result$shape, c(1,0,0))
})

test_that("build_ancestor_matrix_from_outcomes returns empty for Root + Leaf", {

  tree_df <- data.frame(pathString = c("Root/A", "Root/B"))
  tree <- as.Node(tree_df)
  result <- build_ancestor_matrix_from_outcomes(tree)
  expect_equal(result$shape, c(1,0,0))
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
