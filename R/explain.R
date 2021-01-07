
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
#' @examples
#'
#' if (torch::torch_is_installed()) {
#'
#' set.seed(2021)
#'
#' n <- 1000
#' x <- data.frame(
#'   x = runif(n),
#'   y = runif(n),
#'   z = runif(n)
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
#' }
#'
#' @export
tabnet_explain <- function(object, new_data) {
  processed <- hardhat::forge(new_data, object$blueprint)
  data <- resolve_data(processed$predictors, y = data.frame(rep(1, nrow(new_data))))
  output <- explain_impl(object$fit$network, data$x)

  # convert stuff to matrix with colnames
  nms <- colnames(processed$predictors)
  output$M_explain <- convert_to_df(output$M_explain, nms)
  output$masks <- lapply(output$masks, convert_to_df, nms = nms)
  class(output) <- "tabnet_explain"
  output
}

convert_to_df <- function(x, nms) {
  x <- as.data.frame(as.matrix(x$to(device = "cpu")$detach()))
  colnames(x) <- nms
  tibble::as_tibble(x)
}

explain_impl <- function(network, x) {

  outputs <- network$forward_masks(x)

  # summarize the categorical embeddedings into 1 column
  # per variable
  M_explain <- sum_embedding_masks(
    mask = outputs[[1]],
    input_dim = network$input_dim,
    cat_idx = network$cat_idxs,
    cat_emb_dim = network$cat_emb_dim
  )

  masks <- lapply(
    outputs[[2]],
    FUN = sum_embedding_masks,
    input_dim = network$input_dim,
    cat_idx = network$cat_idxs,
    cat_emb_dim = network$cat_emb_dim
  )

  list(M_explain = M_explain, masks = masks)
}

compute_feature_importance <- function(network, x) {
  out <- explain_impl(network, x)
  m <- as.numeric(as.matrix(out$M_explain$sum(dim = 1)$detach()$to(device = "cpu")))
  m/sum(m)
}

# sum embeddings taking their sizes into account.
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

