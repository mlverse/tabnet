test_that("returns correct shape and type for a simple 2-level hierarchy", {
  # Arbre : Root -> {A, B}, A -> {C1, C2}, B -> {D1, D2}
  tree_df <- data.frame(pathString = c(
    "Root/A/C1", "Root/A/C2", "Root/B/D1", "Root/B/D2"
  ))
  tree <- as.Node(tree_df)
  
  outcomes <- tibble::tibble(
    level_2 = factor(c("A", "A", "B", "B")),
    level_3 = factor(c("C1", "C2", "D1", "D2"))
  )
  
  result <- build_ancestor_matrix_from_outcomes(tree, outcomes)
  
  # 2 level_2 classes + 4 level_3 = 6 classes
  expect_tensor(result)
  expect_tensor_shape(result, c(1, 6, 6))
  })

test_that("handles ghost classes (present in tree but absent from outcomes)", {
  # Tree with a "C" branch not in the outcomes
  tree_df <- data.frame(pathString = c(
    "Root/A/C1", "Root/A/C2", 
    "Root/B/D1", "Root/B/D2", 
    "Root/C/E1", "Root/C/E2"
  ))
  tree <- as.Node(tree_df)
  
  # Outcomes shall only contain A and B (C is a "ghost class")
  outcomes <- tibble::tibble(
    level_2 = factor(c("A", "A", "B", "B")), 
    level_3 = factor(c("C1", "C2", "D1", "D2"))
  )
  
  result <- build_ancestor_matrix_from_outcomes(tree, outcomes)
  expect_tensor_shape(result, c(1, 6, 6))
  
  # Check of transitive loop : A(1) is ancestor of C1(3) and C2(4)
  # Order : A(1), B(2), C1(3), C2(4), D1(5), D2(6)
  expect_equal_to_r(result[1, 3, 1], 1) 
  expect_equal_to_r(result[1, 4, 1], 1)
  expect_equal_to_r(result[1, 5, 2], 1) 
  expect_equal_to_r(result[1, 6, 2], 1)
  expect_equal_to_r(result[1, 3, 3], 1) # Self-loop
})

test_that("handles non-unique names across different hierarchy levels", {
  # "Manager" may exist at level_2 (Department) and level_3 (JobRole)
  tree_df <- data.frame(pathString = c(
    "Root/Manager/Rep", 
    "Root/IT/Manager"
  ))
  tree <- as.Node(tree_df)
  
  outcomes <- tibble::tibble(
    level_2 = factor(c("Manager", "IT")),
    level_3 = factor(c("Rep", "Manager"))
  )
  
  result <- build_ancestor_matrix_from_outcomes(tree, outcomes)
  
  # 2 + 2 = 4 classes
  expect_tensor_shape(result, c(1L, 4L, 4L))

  # Order : Manager_lvl2(1), IT(2), Rep(3), Manager_lvl3(4)
  # Manager_lvl2(1) is ancestor of Rep(3)
  expect_equal_to_r(result[1, 3, 1], 1)
  # IT(2) is ancestor of Manager_lvl3(4)
  expect_equal_to_r(result[1, 4, 2], 1)
})

test_that("throws an explicit error when a factor level is missing from the tree", {
  tree_df <- data.frame(pathString = c("Root/A/C1", "Root/A/C2"))
  tree <- as.Node(tree_df)
  
  outcomes <- tibble::tibble(
    level_2 = factor(c("A", "X")), # "X" is not in the tree
    level_3 = factor(c("C1", "C2"))
  )
  
  expect_error(
    build_ancestor_matrix_from_outcomes(tree, outcomes),
    "not found"
  )
})

test_that("throws an error when outcomes contains no factor levels", {
  tree_df <- data.frame(pathString = c("Root/A/C1"))
  tree <- as.Node(tree_df)
  
  outcomes <- tibble::tibble(
    level_2 = factor(character(0)),
    level_3 = factor(character(0))
  )
  
  expect_error(
    build_ancestor_matrix_from_outcomes(tree, outcomes),
    "No factor levels"
  )
})

test_that("preserves the exact class order defined in outcomes factors", {
  # Dans l'arbre, B est défini avant A
  tree_df <- data.frame(pathString = c(
    "Root/B/D1", "Root/B/D2", 
    "Root/A/C1", "Root/A/C2"
  ))
  tree <- as.Node(tree_df)
  
  # Mais dans outcomes, A est explicitement avant B
  outcomes <- tibble::tibble(
    level_2 = factor(c("A", "A", "B", "B"), levels = c("A", "B")),
    level_3 = factor(c("C1", "C2", "D1", "D2"), levels = c("C1", "C2", "D1", "D2"))
  )
  
  result <- build_ancestor_matrix_from_outcomes(tree, outcomes)

  # Order is given by outcomes : A(1), B(2), C1(3), C2(4), D1(5), D2(6)
  # A(1) is ancestor of C1(3) et C2(4)
  expect_equal_to_r(result[1, 3, 1], 1)
  expect_equal_to_r(result[1, 4, 1], 1)
  
  # B(2) is ancestor of D1(5) et D2(6)
  expect_equal_to_r(result[1, 5, 2], 1)
  expect_equal_to_r(result[1, 6, 2], 1)
})

test_that("handles a flat single-level hierarchy correctly", {
  tree_df <- data.frame(pathString = c("Root/A", "Root/B", "Root/C"))
  tree <- as.Node(tree_df)
  
  outcomes <- tibble::tibble(
    level_2 = factor(c("A", "B", "C"))
  )
  
  result <- build_ancestor_matrix_from_outcomes(tree, outcomes)
  expect_equal(result$shape, c(1L, 3L, 3L))
  
  # no hierarchy, only self-loops
  expect_equal_to_r(result$squeeze(1), diag(3))
})

test_that("computes full transitive closure for deep hierarchies (3+ levels)", {
  tree_df <- data.frame(pathString = c(
    "Root/L1_A/L2_A1/L3_A1a", 
    "Root/L1_A/L2_A1/L3_A1b", 
    "Root/L1_A/L2_A2/L3_A2a", 
    "Root/L1_B/L2_B1/L3_B1a"
  ))
  tree <- as.Node(tree_df)
  
  outcomes <- tibble::tibble(
    level_2 = factor(c("L1_A", "L1_A", "L1_A", "L1_B")),
    level_3 = factor(c("L2_A1","L2_A1", "L2_A2", "L2_B1")),
    level_4 = factor(c("L3_A1a", "L3_A1b", "L3_A2a", "L3_B1a"))
  )
  
  result <- build_ancestor_matrix_from_outcomes(tree, outcomes)
  expect_equal(result$shape, c(1L, 9L, 9L))
  
  # Order : L1_A(1), L1_B(2), L2_A1(3), L2_A2(4), L2_B1(5), 
  #         L3_A1a(6), L3_A1b(7), L3_A2a(8), L3_B1a(9)
  
  # L1_A(1) is a transitive ancestor of all the sub-tree A
  expect_equal_to_r(result[1, 3, 1], 1) # -> L2_A1
  expect_equal_to_r(result[1, 6, 1], 1) # -> L3_A1a
  expect_equal_to_r(result[1, 8, 1], 1) # -> L3_A2a
  
  # L2_A1(3) is an ancestor of all its direct children
  expect_equal_to_r(result[1, 6, 3], 1) # -> L3_A1a
  expect_equal_to_r(result[1, 7, 3], 1) # -> L3_A1b
  
  # Self-loops on the diagonal (substracting the eye don't go to negative values)
  expect_true((result$squeeze() - torch::torch_eye(9))$min()$item() >= 0)
})