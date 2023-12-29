
#' Interpretation metrics from a TabNet model
#'
#' @param object a TabNet fit object
#' @param new_data a data.frame to obtain interpretation metrics.
#'
#' @return
#'
#' Returns a list with
#'
#' * `M_explain`: the aggregated feature importance masks as detailed in
#'   TabNet's paper.
#' * `masks` a list containing the masks for each step.
#'
#' @examplesIf torch::torch_is_installed()
#'
#' set.seed(2021)
#'
#' n <- 1000
#' x <- data.frame(
#'   x = rnorm(n),
#'   y = rnorm(n),
#'   z = rnorm(n)
#' )
#'
#' y <- x$x
#'
#' fit <- tabnet_fit(x, y, epochs = 20,
#'                   num_steps = 1,
#'                   batch_size = 512,
#'                   attention_width = 1,
#'                   num_shared = 1,
#'                   num_independent = 1)
#'
#'
#'  ex <- tabnet_explain(fit, x)
#'
#'
#' @export
tabnet_explain <- function(object, new_data) {
  UseMethod("tabnet_explain")
}

#' @export
#' @rdname tabnet_explain
tabnet_explain.default <- function(object, new_data) {
  stop(domain=NA,
       gettextf("`tabnet_explain()` is not defined for a '%s'.", class(object)[1]),
       call. = FALSE)
}

#' @export
#' @rdname tabnet_explain
tabnet_explain.tabnet_fit <- function(object, new_data) {
  if (inherits(new_data, "Node")) {
    new_data_df <- node_to_df(new_data)$x
  } else {
    new_data_df <- new_data
  }
  # Enforces column order, type, column names, etc
  processed <- hardhat::forge(new_data_df, object$blueprint, outcomes = FALSE)
  data <- resolve_data(processed$predictors, y = rep(1, nrow(processed$predictors)))
  device <- get_device_from_config(object$fit$config)
  data <- to_device(data, device)
  output <- explain_impl(object$fit$network, data$x, data$x_na_mask)

  # convert stuff to matrix with colnames
  nms <- colnames(processed$predictors)
  output$M_explain <- convert_to_df(output$M_explain, nms)
  output$masks <- lapply(output$masks, convert_to_df, nms = nms)
  class(output) <- "tabnet_explain"
  output
}

#' @export
#' @rdname tabnet_explain
tabnet_explain.tabnet_pretrain <- tabnet_explain.tabnet_fit

#' @export
#' @rdname tabnet_explain
tabnet_explain.model_fit <- function(object, new_data) {
  tabnet_explain(parsnip::extract_fit_engine(object), new_data)
}

convert_to_df <- function(x, nms) {
  x <- as.data.frame(as.matrix(x$to(device = "cpu")$detach()))
  colnames(x) <- nms
  tibble::as_tibble(x)
}

#' @importFrom purrr map map2
#' @importFrom stats cor
explain_impl <- function(network, x, x_na_mask, with_stability = FALSE) {
  curr_device <- network$.check$device
  withr::defer({
    network$to(device = curr_device)
  })
  network$to(device=x$device)
  # NULLing values to avoid a R-CMD Check Note "No visible binding for global variable"
  M_explain_emb_dim <- masks_emb_dim <- NULL
  c(M_explain_emb_dim, masks_emb_dim) %<-% network$forward_masks(x, x_na_mask)

  if (with_stability) {
    # TODO
    # Compute InterpreStability value through 5-group, the lazy way
    # define 5 sampled id groups
    obs_group <- lapply(1:5, FUN = sample.int, n = nrow(x), size = ceiling(nrow(x)/5))
    x_group <- map(obs_group, ~x[.x,] )
    x_na_mask_group <- map(obs_group, ~x_na_mask[.x,] )

    M_explain_mask_lst <- map2(x_group, x_na_mask_group, network$forward_masks)
    M_explain <- map(M_explain_mask_lst, ~sum_embedding_masks(
      mask = .x[[1]],
      input_dim = network$input_dim,
      cat_idx = network$cat_idxs,
      cat_emb_dim = network$cat_emb_dim
    )
    )
    m <- map(M_explain, ~as.numeric(as.matrix(.x$sum(dim = 1)$detach()$to(device = "cpu"))))
    compute_feature_importance <-  map(m, ~.x/sum(.x))

    corr_mat <- torch::torch_stack(compute_feature_importance, dim = 1) |> as.matrix() |> t() |> cor(method = "pearson")
    corr_mat <- corr_mat * (torch::torch_ones_like(corr_mat) - torch::torch_eye(n = dim(corr_mat)[1] ))
    interprestability <- (corr_mat[corr_mat >=0.9]$sum() +
      .8 * corr_mat[corr_mat < 0.9 & corr_mat >= 0.7]$sum() +
      .6 * corr_mat[corr_mat < 0.7 & corr_mat >= 0.5]$sum() +
      .4 * corr_mat[corr_mat < 0.5 & corr_mat >= 0.3]$sum() +
      .2 * corr_mat[corr_mat < 0.3]$sum()) / (corr_mat$shape[1] * (corr_mat$shape[2] - 1))

  } else {
    interprestability <- NULL
  }
  # summarize the categorical embedding into 1 column
  # per variable
  M_explain <- sum_embedding_masks(
    mask = M_explain_emb_dim,
    input_dim = network$input_dim,
    cat_idx = network$cat_idxs,
    cat_emb_dim = network$cat_emb_dim
  )

  masks <- lapply(
    masks_emb_dim,
    FUN = sum_embedding_masks,
    input_dim = network$input_dim,
    cat_idx = network$cat_idxs,
    cat_emb_dim = network$cat_emb_dim
  )

  list(M_explain = M_explain$to(device="cpu"),
       masks = to_device(masks, "cpu"),
       interprestability = interprestability)
}

compute_feature_importance <- function(network, x, x_na_mask) {
  out <- explain_impl(network, x, x_na_mask)
  m <- as.numeric(as.matrix(out$M_explain$sum(dim = 1)$detach()$to(device = "cpu")))
  list(importance = m/sum(m), interprestability = out$interprestability)
}

# sum embeddings, taking their sizes into account.
sum_embedding_masks <- function(mask, input_dim, cat_idx, cat_emb_dim) {
  sizes <- rep(1, input_dim)
  sizes[cat_idx] <- cat_emb_dim

  splits <- mask$split_with_sizes(sizes, dim = 2)
  splits <- lapply(splits, torch::torch_sum, dim = 2, keepdim = TRUE)

  torch::torch_cat(splits, dim = 2)
}

vi_model.tabnet_fit <- function(object, ...) {
  tib <- object$fit$importances
  names(tib) <- c("Variable", "Importance")
  tib
}

vi_model.tabnet_pretrain <- vi_model.tabnet_fit
