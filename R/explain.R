
#' Interpretation metrics from a TabNet model
#'
#' @param object a TabNet fit object
#' @param new_data a data.frame to obtain interpretation metrics.
#' @param stability shall we compute the InterpreStability score of the model with
#'  it's last 5 checkpoints?.
#'
#' @return
#'
#' Returns a list with
#'
#' * `M_explain`: the aggregated feature importance masks as detailed in
#'   TabNet's paper.
#' * `masks` a list containing the masks for each step.
#' * `interprestability` The InterpreStability score of model with its last 5 checkpoints.
#'  according to Interpretabnet paper, the score represent the stability of feature importance mask:
#'  • InterpreStability ∈ [0.9, 1]: Very high stability
#'  • InterpreStability ∈ [0.7, 0.9]: High stability
#'  • InterpreStability ∈ [0.5, 0.7]: Moderate stability
#'  • InterpreStability ∈ [0.3, 0.5]: Low stability
#'  • InterpreStability ∈ [0, 0.3]: Little if any stability
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
#'  ex$interprestability
#'
#' @export
tabnet_explain <- function(object, new_data, stability) {
  UseMethod("tabnet_explain")
}

#' @export
#' @rdname tabnet_explain
tabnet_explain.default <- function(object, new_data, stability) {
  stop(domain=NA,
       gettextf("`tabnet_explain()` is not defined for a '%s'.", class(object)[1]),
       call. = FALSE)
}

#' @export
#' @rdname tabnet_explain
tabnet_explain.tabnet_fit <- function(object, new_data, stability = TRUE) {
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

  # Compute InterpreStability value through last 5 checkpoints
  n_checkpoints <- length(object$fit$checkpoints)
  if (stability && n_checkpoints > 5 ) {
    # TODO
    # Compute feature importance value through last 5 checkpoints
    computed_feature_importance <- purrr::map(
      object$fit$checkpoints[(n_checkpoints-5):n_checkpoints],
      ~compute_feature_importance(reload_model(.x), data$x, data$x_na_mask),
      .progress = "InterpreStability score within last 6 models")
    #
    corr_mat <- data.frame(computed_feature_importance) |> set_names(as.character(seq(-5,0))) |> cor(method = "pearson")
    diag(corr_mat) <- 0
    interprestability <- (sum(corr_mat[corr_mat >= 0.9]) +
                            .8 * sum( corr_mat[corr_mat < 0.9 & corr_mat >= 0.7]) +
                            .6 * sum( corr_mat[corr_mat < 0.7 & corr_mat >= 0.5]) +
                            .4 * sum( corr_mat[corr_mat < 0.5 & corr_mat >= 0.3]) +
                            .2 * sum( corr_mat[corr_mat < 0.3])) / (prod(dim(corr_mat)) - dim(corr_mat)[1])
  } else {
    interprestability <- NULL
  }


  # convert stuff to matrix with colnames
  nms <- colnames(processed$predictors)
  output$M_explain <- convert_to_df(output$M_explain, nms)
  output$masks <- lapply(output$masks, convert_to_df, nms = nms)
  output$interprestability <- interprestability
  class(output) <- "tabnet_explain"
  output
}

#' @export
#' @rdname tabnet_explain
tabnet_explain.tabnet_pretrain <- tabnet_explain.tabnet_fit

#' @export
#' @rdname tabnet_explain
tabnet_explain.model_fit <- function(object, new_data, stability) {
  tabnet_explain(parsnip::extract_fit_engine(object), new_data, stability)
}

convert_to_df <- function(x, nms) {
  x <- as.data.frame(as.matrix(x$to(device = "cpu")$detach()))
  colnames(x) <- nms
  tibble::as_tibble(x)
}

explain_impl <- function(network, x, x_na_mask, with_stability = FALSE) {
  curr_device <- network$.check$device
  withr::defer({
    network$to(device = curr_device)
  })
  network$to(device=x$device)
  # NULLing values to avoid a R-CMD Check Note "No visible binding for global variable"
  M_explain_emb_dim <- masks_emb_dim <- NULL
  c(M_explain_emb_dim, masks_emb_dim) %<-% network$forward_masks(x, x_na_mask)

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
       masks = to_device(masks, "cpu")
       )
}

compute_feature_importance <- function(network, x, x_na_mask) {
  out <- explain_impl(network, x, x_na_mask)
  m <- as.numeric(as.matrix(out$M_explain$sum(dim = 1)$detach()$to(device = "cpu")))
  m/sum(m)
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
