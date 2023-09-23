#' Create the group matrix corresponding to the given vars_id_groups
#'
#' @param vars_id_groups : list of vars_id_group, vars_id_group being the vector of features ids to group
#'     Each element represents features in the same group. Feature id cannot be present in
#'     multiple group, ids that are not part of a group will be implicitly part of a single feature group.
#' @param input_dim : number of feature in the initial dataset
#'
#' @return a torch_tensor of size (n_groups, input_dim)
#'     where m_ij represents the importance of feature j in group i
#'     The row sum must be 1 as each group is equally important a priori.
#'
#' @examplesIf torch::torch_is_installed()
#' # mtcars example
#' colnames(mtcars)
#' create_group_matrix(vars_id_groups = list(c(1,3), c(4:7), c(9:10)), input_dim = 11)
create_group_matrix <- function(vars_id_groups, input_dim) {
  # TODO shall be secured and simplified with sparse matrix functions
  check_vars_id_groups(vars_id_groups, input_dim)

  if (length(vars_id_groups) == 0) {
    group_matrix <- torch::torch_eye(input_dim)
    return(group_matrix)
  } else {
    n_groups <- input_dim - sum(purrr::map_dbl(vars_id_groups, length)) + length(vars_id_groups)
    group_matrix <- torch::torch_zeros(c(n_groups, input_dim))

    remaining_features <- seq(input_dim)

    current_group_idx <- 1
    for (group in vars_id_groups) {
      group_size <- length(group)
      for (feature_id in group) {
        # add importance of element in group matrix and corresponding group
        group_matrix[current_group_idx, feature_id] <- 1 / group_size
        # remove features from list of features
      }
      remaining_features <- setdiff(remaining_features, group)
      # move to next group
      current_group_idx = current_group_idx + 1
    }
    # features not mentionned in vars_id_groups get assigned their own group of singleton
    for (remaining_feat_idx in remaining_features) {
      group_matrix[current_group_idx, remaining_feat_idx] <- 1
      current_group_idx = current_group_idx + 1
    }
    return(group_matrix)
  }
}

#' An eye tensor shrinking the post_embedding tensor into originel tensor
#'  in order to sum importances related to the same categorical embedding back to
#'  the initial index.
#' TODO very close to `create_explain_matrix` and to `create_group_matrix` and shall be factorized
#' translated from utils.create_explain_matrix()
#'
#' @param input_dim :  Initial input dim
#' @param cat_emb_dim : size of embedding for all (like 3 ) or each (like in c(3,2,1,3) )
#'  of the categorical feature
#' @param cat_idx : position of categorical features
#' @param post_embed_dim : int Post-embedding number of features
#'
#' @eturn  reducing_matrix of dim (post_embed_dim, input_dim)  to perform explain matrix reduction
#' @noRd
#' @examplesIf torch::torch_is_installed()
#' # starwars example
#' data(storms, package="dplyr")
#' cat_emb_dim <- c(3,2)
#' cat_idx <- which(!sapply(storms, is.numeric))
#' create_explain_matrix(input_dim = 13, cat_emb_dim, cat_idx)
create_explain_matrix <- function(input_dim, cat_emb_dim, cat_idx) {
  # record continuous indices
  numerical_idx <- !seq(1,input_dim) %in% cat_idx
  sizes <- rep(1, length(numerical_idx))
  sizes[!numerical_idx] <- cat_emb_dim

  eye_split <- torch::torch_eye(input_dim)$split(1, dim = 1)
  # use torch broadcasting for 3x performance
  # reducing_matrix <- purrr::map2(eye_split, sizes, ~.x$mul(torch::torch_ones(c(.y, 1))))
  reducing_matrix <- purrr::map2(eye_split, sizes, ~.x$broadcast_to(c(.y, input_dim)))

  return(torch::torch_cat(reducing_matrix, dim = 1))
}


#' expand the x boolean tensor to its size after embedding
#'
#' extracted from tabnet::tab-network::na_embedding_generator()
#' TODO use boolean operation to lower footprint
#' @param x boolean tensor
#' @param numerical_idx a boolean vector of the numerical features of x
#' @param cat_emb_dim the embedding dimension of each categorical feature of x
#'
#' @return a boolean torch_tensor of size [length(numerical_idx), sum(cat_emb_dim)]
#' @noRd
#' @examples
embedding_expander <- function(x, numerical_idx, cat_emb_dim) {

  nrows <- x$shape[1]
  sizes <- rep(1, length(numerical_idx))
  sizes[!numerical_idx] <- cat_emb_dim

  splits <- x$split(1, dim = 2)
  #use torch broadcasting for 3x performance
  # splits <- purrr::map2(splits, sizes, ~.x$mul(torch::torch_ones(c(1, .y))$to(torch::torch_bool())))
  splits <- purrr::map2(splits, sizes, ~.x$broadcast_to(c(nrows, .y)))

  # concat
  torch::torch_cat(splits, dim = 2)
}

#' Check that vars_id_groups:
#'     - is a list of vectors
#'     - does not contain twice the same feature in different groups
#'     - does not contain unknown features (>= input_dim)
#'     - does not contain empty groups
#' @param vars_id_groups : list of vars_id_group, vars_id_group being the vector of features ids to group
#' @param input_dim : number of feature in the initial dataset
#' @noRd
check_vars_id_groups <- function(vars_id_groups, input_dim) {
  if (!is.list(vars_id_groups)) {
    rlang::abort(glue::glue("`vars_id_groups` must be a list."))
  }
  if (length(vars_id_groups) == 0) {
    return
  } else {
    # ensure groups are not empty
    bad_group_ids <- purrr::map_lgl(vars_id_groups, ~(!is.vector(.x) | !length(.x) > 0))
    if (any(bad_group_ids)) {
      rlang::abort(glue::glue("Each vars_id_group must be a non empty vector. This fails with {vars_id_groups[bad_group_ids]}"))
    }
  }
  # ensure there is no id overlap
  all_ids <- unlist(vars_id_groups)
  if (any(duplicated(all_ids))) {
    rlang::abort(glue::glue("`vars_id` {all_ids[duplicated(all_ids)]} appears more than once in the `vars_id_groups`"))
  }
  # ensure all ids are within input_dim
  if (any((all_ids > input_dim) | (all_ids < 1) )) {
    rlang::abort(glue::glue("`vars_id` {all_ids[(all_ids > input_dim) | (all_ids < 1) ]} are wrong ids for an `input_dim` of {input_dim}"))
  }
}

#' Check consistency of parameters related to categorical embeddings
#'   expand cat_emb_dim and sort all in ascending ids
#' @noRd
check_embedding_parameters <- function(cat_dims, cat_idx, cat_emb_dim){
  if (length(cat_dims) == 0 | length(cat_idx) == 0) {
    rlang::abort(glue::glue("`cat_dims` and `cat_idx` cannot be null"))
  }
  if (length(cat_dims) != length(cat_idx)) {
    rlang::abort(glue::glue("`cat_dims` and `cat_idx` must have the same length."))
  }

  if (length(cat_emb_dim) == 1) {
    cat_emb_dims <- rep(cat_emb_dim, length(cat_idx))
  } else {
    cat_emb_dims <- cat_emb_dim
  }

  # check that all embeddings are provided
  if (length(cat_emb_dims) != length(cat_dims)) {
    rlang::abort(glue::glue("`cat_emb_dim` length must be 1 or the number of categorical predictors, got length {length(cat_emb_dims)} for {length(cat_dims)} categorical predictors"))
  }

  # Rearrange to get reproducible seeds with different ordering
  cat_idx_df <- data.frame(cat_idx = cat_idx, row_id = seq_along(cat_idx))
  cat_id_sorted <- cat_idx_df[order(cat_idx_df$cat_idx),]$row_id
  cat_idx_sorted <- cat_idx[cat_id_sorted]
  cat_dims <- cat_dims[cat_id_sorted]
  cat_emb_dims <- cat_emb_dims[cat_id_sorted]

  return(list(cat_dims, cat_idx_sorted, cat_emb_dims))
}
