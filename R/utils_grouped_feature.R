#' Create the group matrix corresponding to the given vars_id_groups
#'
#' @vars_id_groups : list of vars_id_group, vars_id_group being the vector of features ids to group
#'     Each element represents features in the same group. Feature id cannot be present in
#'     multiple group, ids that are not part of a group will be implicitly part of a single feature group.
#' @input_dim : number of feature in the initial dataset
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
#' Check that vars_id_groups:
#'     - is a list of list
#'     - does not contain twice the same feature in different groups
#'     - does not contain unknown features (>= input_dim)
#'     - does not contain empty groups
#' @vars_id_groups : list of vars_id_group, vars_id_group being the vector of features ids to group
#' @input_dim : number of feature in the initial dataset
check_vars_id_groups <- function(vars_id_groups, input_dim) {
    if (!is.list(vars_id_groups)) {
      rlang::abort(glue::glue("vars_id_groups must be a list."))
    }
    if (length(vars_id_groups) == 0) {
      return
    } else {
      # ensure groups are not empty
      for (group in vars_id_groups) {
          if (!is.vector(group) || !length(group) > 0) {
            rlang::abort(glue::glue("each vars_id_group must be a non empty vector. This fails with {group}"))
          }
      }
    }
  # ensure there is no id overlap
  all_ids <- unlist(vars_id_groups)
  if (any(duplicated(all_ids))) {
    rlang::abort(glue::glue("vars_id {all_ids[duplicated(all_ids)]} appears more than once in the `vars_id_groups`"))
  }
  # ensure all ids are within input_dim
  if (any((all_ids > input_dim) | (all_ids < 1) )) {
    rlang::abort(glue::glue("vars_id {all_ids[(all_ids > input_dim) || (all_ids < 1) ]} are wrong ids for an `input_dim` of {input_dim}"))
  }
}


# Check parameters related to embeddings and rearrange them in a unique manner.
check_embedding_parameters <- function(cat_dims, cat_idxs, cat_emb_dim){
    if (length(cat_dims) == 0 || length(cat_idxs) == 0) {
      rlang::abort(glue::glue("`cat_dims` and `cat_idxs` cannot be null"))
    }
  if (length(cat_dims) != length(cat_idxs)) {
    rlang::abort(glue::glue("`cat_dims` and `cat_idxs` must have the same length."))
    }

    if (is.integer(cat_emb_dim)) {
        cat_emb_dims <- cat_emb_dim * length(cat_idxs)
    } else {
        cat_emb_dims <- cat_emb_dim
    }

    # check that all embeddings are provided
    if (length(cat_emb_dims) != length(cat_dims)) {
        rlang::abort(glue::glue("`cat_emb_dim` and `cat_dims` must have the same length, got {length(cat_emb_dims)} and {length(cat_dims)}"))
    }

    # Rearrange to get reproducible seeds with different ordering
      cat_dims <- cat_dims[cat_idxs]
      cat_emb_dims <- cat_emb_dims[cat_idxs]

    return(list(cat_dims, cat_idxs, cat_emb_dims))
