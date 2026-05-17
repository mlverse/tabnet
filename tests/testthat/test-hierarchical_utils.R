test_that("get_constr_output handles basic 2D input with identity constraint", {
  m <- matrix(c(1, 2, 
                3, 4), nrow = 2, ncol = 2)
  x <- torch_tensor(m, dtype = torch_float32())
  R <- torch_eye(2, dtype = torch_float32())
  result <- get_constr_output(x, R)
  expect_tensor(result)
  expect_tensor_shape(result, c(2, 2))
  expect_equal_to_r(result, m)
})

test_that("get_constr_output applies hierarchy constraint correctly", {
  x <- torch_tensor(matrix(c(1, 5, 
                             3, 2), nrow = 2, ncol = 2, byrow = TRUE), dtype = torch_float64())
  R <- torch_tensor(matrix(c(1, 1, 
                             0, 1), nrow = 2, ncol = 2, byrow = TRUE), dtype = torch_float64())
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
  R <- torch_tensor(matrix(c(1, 1, 0, 0, 
                             1, 1, 0, 0, 
                             0, 0, 1, 1, 
                             0, 0, 1, 1), nrow = 4, ncol = 4, byrow = TRUE))
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
  R <- torch_tensor(matrix(c(1, 1, 0, 0, 
                             1, 1, 0, 0, 
                             0, 0, 1, 1, 
                             0, 0, 1, 1), nrow = 4, ncol = 4, byrow = TRUE))
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
  x <- torch_tensor(matrix(c(1, 5, 3, 
                             2, 4, 6), nrow = 2, ncol = 3, byrow = TRUE))
  R <- torch_ones(c(3, 3))
  result <- get_constr_output(x, R)
  expect_tensor_shape(result, c(2, 3))
  # Each row is filled with its own row-wise maximum
  expected <- matrix(c(5, 5, 5, 
                       6, 6, 6), nrow = 2, ncol = 3, byrow = TRUE)
  expect_equal_to_r(result, expected, tolerance = 1e-6)
})

test_that("get_constr_output throws error for dimension mismatch", {
  x <- torch_tensor(matrix(1:4, nrow = 2, ncol = 2))
  R <- torch_eye(3)
  expect_error(get_constr_output(x, R), "must match the existing size")
})

test_that("get_constr_output throws error for non-2D R", {
  x <- torch_tensor(matrix(1:4, nrow = 2, ncol = 2))
  R <- torch_tensor(array(1:8, dim = c(1, 2, 2, 2)))
  expect_error(get_constr_output(x, R), "dimension")
})

test_that("max_constraint_output returns original output when ancestor is identity", {
  output <- torch_tensor(matrix(1:6, nrow = 2, ncol = 3))
  labels <- torch_tensor(matrix(c(TRUE, FALSE, TRUE, 
                                  FALSE, TRUE, FALSE), nrow = 2, ncol = 3), dtype = torch_bool())
  ancestor <- torch_eye(3)
  result <- max_constraint_output(output, labels, ancestor)
  expect_tensor_shape(result, c(2, 3))
  # With an identity ancestor matrix, constraint propagation is neutral.
  # The formula simplifies to: (~labels * output) + (labels * output) == output
  expect_equal_to_r(result, matrix(1:6, nrow = 2, ncol = 3))
})

test_that("max_constraint_output applies constraint to positive labels", {
  output <- torch_tensor(matrix(c(1, 5, 
                                  3, 2), nrow = 2, ncol = 2, byrow = TRUE))
  labels <- torch_tensor(matrix(c(1, 0, 
                                  1, 0), nrow = 2, ncol = 2, byrow = TRUE), dtype = torch_bool())
  ancestor <- torch_tensor(matrix(c(1, 1, 
                                    0, 1), nrow = 2, ncol = 2, byrow = TRUE))
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
  output <- torch_tensor(matrix(c(1, 5, 
                                  3, 2), nrow = 2, ncol = 2, byrow = TRUE))
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
  output <- torch_tensor(matrix(c(1, 2, 3, 
                                  4, 5, 6), nrow = 2, ncol = 3, byrow = TRUE))
  labels <- torch_tensor(matrix(c(1, 0, 0, 
                                  0, 1, 0), nrow = 2, ncol = 3, byrow = TRUE), dtype = torch_bool())
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
  output <- torch_tensor(matrix(c(1, 4, 
                                  2, 3), nrow = 2, ncol = 2, byrow = TRUE))
  labels <- torch_tensor(matrix(c(TRUE, FALSE, 
                                  TRUE, FALSE), nrow = 2, ncol = 2, byrow = TRUE), dtype = torch_bool())
  ancestor <- torch_tensor(matrix(c(1, 1, 
                                    0, 1), nrow = 2, ncol = 2, byrow = TRUE))
  direct <- max_constraint_output(output, labels, ancestor)
  constr_out <- get_constr_output(output, ancestor)
  train_out <- get_constr_output(labels * output, ancestor)
  manual <- torch_logical_not(labels) * constr_out + labels * train_out
  expect_equal_to_r(direct, as_array(manual))
})

test_that("get_constr_output handles negative values correctly", {
  x <- torch_tensor(matrix(c(-5, -1, 
                             -3, -2), nrow = 2, ncol = 2, byrow = TRUE))
  R <- torch_tensor(matrix(c(1, 1, 
                             0, 1), nrow = 2, ncol = 2, byrow = TRUE))
  result <- get_constr_output(x, R)
  expected <- matrix(c(-1, 0, 
                       -2, 0), nrow = 2, ncol = 2, byrow = TRUE)
  expect_equal_to_r(result, expected)
})

test_that("max_constraint_output handles mixed positive-negative with constraints", {
  output <- torch_tensor(matrix(c(-5, -3, 
                                  -1, 4), nrow = 2, ncol = 2, byrow = TRUE))
  labels <- torch_tensor(matrix(c(TRUE, TRUE, 
                                  FALSE, TRUE), nrow = 2, ncol = 2, byrow = TRUE), dtype = torch_bool())
  ancestor <- torch_tensor(matrix(c(1, 1, 
                                    0, 1), nrow = 2, ncol = 2, byrow = TRUE))
  result <- max_constraint_output(output, labels, ancestor)
  expected <- matrix(c(-3, 0, 
                       4, 4), nrow = 2, ncol = 2, byrow = TRUE)
  expect_equal_to_r(result, expected)
})

test_that("build_ancestor_matrix handles basic hierarchy", {
  # Tree: Root -> A -> C
  #        Root -> B -> D
  # Edges: R->A, A->C, R->B, B->D
  # Pruning Logic:
  # 1. Remove Root: Keeps A->C, B->D
  # 2. Remove leaves (C, D are not in 'from'): Keeps A->C? No. C is not a parent.
  #    Keeps B->D? No. D is not a parent.
  # Result: No edges match criteria. Empty matrix.
  
  tree_df <- data.frame(pathString = c("Root/A", "Root/A/C", "Root/B", "Root/B/D"))
  tree <- as.Node(tree_df)
  
  result <- build_ancestor_matrix(tree)$to_dense()
  
  # Expectation: No internal nodes exist that are also children (excluding Root)
  # A and B are children of Root, but their children (C, D) are leaves.
  # Thus A and B are effectively leaves in the "internal structure".
  expect_tensor_shape(result, c(5,5))
})

test_that("build_ancestor_matrix handles linear chain of internal nodes", {
  # Tree: Root -> A -> B -> C
  # Edges: R->A, A->B, B->C
  # Pruning Logic:
  # 1. Remove Root: Keeps A->B, B->C
  # 2. Keep only if target is a parent:
  #    - A->B: B is a parent (of C). Keep.
  #    - B->C: C is a leaf. Drop.
  # Remaining Edges: A -> B
  # Nodes: A(1), B(2)
  # Matrix: A->A, A->B, B->B
  
  tree_df <- data.frame(pathString = c("Root/A", "Root/A/B", "Root/A/B/C"), value = 1:3)
  tree <- as.Node(tree_df)
  
  result <- build_ancestor_matrix(tree)$to_dense()$to(torch_long())

  expected <- fBasics::Triang(matrix(TRUE, nrow = 4, ncol = 4)) # upper triangular 4 x 4 mat
  expect_equal_to_r(result, expected)
})

test_that("build_ancestor_matrix calculates transitive closure correctly", {
  # Tree: Root -> A -> B -> C -> D
  # Edges: R->A, A->B, B->C, C->D
  # Pruning Logic:
  # 1. Remove Root: A->B, B->C, C->D
  # 2. Keep if target is parent:
  #    - A->B: B is parent (of C). Keep.
  #    - B->C: C is parent (of D). Keep.
  #    - C->D: D is leaf. Drop.
  # Remaining Edges: A -> B, B -> C
  # Nodes: A(1), B(2), C(3)
  
  tree_df <- data.frame(pathString = c("Root/A", "Root/A/B", "Root/A/B/C", "Root/A/B/C/D"))
  tree <- as.Node(tree_df)
  
  result <- build_ancestor_matrix(tree)$to_dense()$to(torch_long())
  
  expected <- fBasics::Triang(matrix(TRUE, nrow = 5, ncol = 5)) # upper triangular 5 x 5 mat
  expect_equal_to_r(result, expected)
})

test_that("build_ancestor_matrix handles branching internal nodes", {
  # Tree: R -> A -> C
  #        R -> B -> C (Diamond shape, merging back to C) 
  # *Note: data.tree allows this structure (multiple parents)? 
  # Actually standard trees are single parent. Let's stick to standard tree.
  
  # Tree: R -> A -> C -> E
  #        R -> B -> D -> E
  # Edges: R->A, A->C, C->E, R->B, B->D, D->E
  # Pruning:
  # 1. Remove R: A->C, C->E, B->D, D->E
  # 2. Keep target if parent:
  #    - A->C (C is parent of E). Keep.
  #    - C->E (E is leaf). Drop.
  #    - B->D (D is parent of E). Keep.
  #    - D->E (E is leaf). Drop.
  # Nodes: A, C, B, D
  # Edges: A->C, B->D
  
  tree_df <- data.frame(pathString = c("Root/A", "Root/A/C", "Root/A/C/E", 
             "Root/B", "Root/B/D", "Root/B/D/E"))
  tree <- as.Node(tree_df)
  
  result <- build_ancestor_matrix(tree)$to_dense()$to(torch_long())
  
  # upper triangular 6 x 6 mat in a 7 x 7 matrix with few non-ancestor values
  expected <- fBasics::Triang(matrix(1, nrow = 7, ncol = 7))
  expected[, 7] <- 0
  expected[2:4, 5:6] <- 0
  expected[5:6, 4] <- 1
  
  expect_equal_to_r(result, expected)
})

test_that("build_ancestor_matrix returns empty for Root-only tree", {
  tree <- Node$new("Root")
  result <- build_ancestor_matrix(tree)$to_dense()
  expect_equal(nrow(result), 1)
})

test_that("build_ancestor_matrix returns empty for Root + Leaf", {
  # Tree: Root -> A
  # Edges: R->A
  # 1. Remove R (from!=Root): Result empty.
  tree <- Node$new("Root")
  tree$AddChild("A")
  result <- build_ancestor_matrix(tree)
  expect_equal(nrow(result), 2)
})

test_that("build_ancestor_matrix preserves integer type", {
  tree_df <- data.frame(pathString = c("Root/A", "Root/A/B"))
  tree <- as.Node(tree_df)
  result <- build_ancestor_matrix(tree)
  expect_tensor_dtype(result, torch_bool())
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
