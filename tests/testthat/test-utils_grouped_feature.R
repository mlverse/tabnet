test_that("create_group_matrix works", {
  vars_id_groups <- list(c(1, 3), c(4:7), c(9:10))
  expect_no_error(create_group_matrix(vars_id_groups, 11))
  expect_tensor_shape(create_group_matrix(vars_id_groups, 11), c(6,11))
  # rowSums must always be 1
  expect_equal_to_r(create_group_matrix(vars_id_groups, 11)$sum(dim = 2),
                    rep(1,6))
})

test_that("create_group_matrix works with no group", {
  vars_id_groups <- list()
  expect_no_error(create_group_matrix(vars_id_groups, 11))
  expect_tensor_shape(create_group_matrix(vars_id_groups, 11), c(11,11))
  expect_equal_to_tensor(create_group_matrix(vars_id_groups, 11),
                         torch::torch_eye(11))
})

test_that("create_group_matrix detects errors in vars id", {
  vars_id_groups <- list(c(1, 4), c(4:7), c(9:10))
  expect_error(create_group_matrix(vars_id_groups, 11),
               "more than once")
  vars_id_groups <- list(c(1, 3), c(4:7), c(9:13))
  expect_error(create_group_matrix(vars_id_groups, 11),
               "wrong ids")
  vars_id_groups <- list(c(), c(4:7), c(9:10))
  expect_error(create_group_matrix(vars_id_groups, 11),
               "non empty")
})

test_that("check_embedding_parameters works", {
  # using the ames dataset
  cat_idx <- which(sapply(ames, is.factor))
  cat_dims <- sapply(cat_idx, function(i) nlevels(ames[[i]]))
  # constant embedding dim
  cat_emb_dim <- 3
  expect_no_error(check_embedding_parameters(cat_dims, cat_idx, cat_emb_dim))
  result_embedding_parameters <- check_embedding_parameters(cat_dims, cat_idx, cat_emb_dim)
  expect_type(result_embedding_parameters, "list")
  expect_length(result_embedding_parameters, 3)
  expect_equal(result_embedding_parameters[[1]], cat_dims)
  expect_equal(result_embedding_parameters[[2]], rep(3, length(cat_idx)))
  expect_equal(result_embedding_parameters[[3]], cat_emb_dim)
  # tailored embedding dim
  cat_emb_dim <- sample.int(5, size = length(cat_idx), replace = TRUE)
  expect_no_error(check_embedding_parameters(cat_dims, cat_idx, cat_emb_dim))
  result_embedding_parameters <- check_embedding_parameters(cat_dims, cat_idx, cat_emb_dim)
  expect_type(result_embedding_parameters, "list")
  expect_length(result_embedding_parameters, 3)
  expect_equal(result_embedding_parameters[[1]], cat_dims)
  expect_equal(result_embedding_parameters[[2]], cat_idx)
  expect_equal(result_embedding_parameters[[3]], cat_emb_dim)
})

test_that("check_embedding_parameters detects categorical inconsistencies", {
  # using the ames dataset
  cat_idx <- which(sapply(ames, is.factor))
  cat_dims <- sapply(cat_idx, function(i) nlevels(ames[[i]]))
  cat_emb_dim <- sample.int(5, size = length(cat_idx), replace = TRUE)
  expect_error(check_embedding_parameters(cat_dims, NULL, cat_emb_dim),
               "cannot be null")
  expect_error(check_embedding_parameters(NULL, cat_idx, cat_emb_dim),
               "cannot be null")
  expect_error(check_embedding_parameters(cat_dims, cat_idx, NULL),
               "cannot be null")
  expect_error(check_embedding_parameters(cat_dims[1:10], cat_idx, cat_emb_dim),
               " must have the same length")
  expect_error(check_embedding_parameters(cat_dims, cat_idx[1:10], cat_emb_dim),
               " must have the same length")
  expect_error(check_embedding_parameters(cat_dims, cat_idx, cat_emb_dim[1:10]),
               " must have the same length")
})

test_that("check_embedding_parameters can reorder categorical ids", {
  # using the ames dataset
  cat_idx <- which(sapply(ames, is.factor)) %>% sample(40)
  cat_dims <- sapply(cat_idx, function(i) nlevels(ames[[i]]))
  cat_emb_dim <- sample.int(5, size = length(cat_idx), replace = TRUE)
  expect_no_error(check_embedding_parameters(cat_dims, cat_idx, cat_emb_dim))
  result_embedding_parameters <- check_embedding_parameters(cat_dims, cat_idx, cat_emb_dim)
  expect_equal(result_embedding_parameters[[2]], sort(cat_idx))
})


