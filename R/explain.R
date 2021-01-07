
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
#'
#'
tabnet_explain <- function(object, new_data) {
  processed <- hardhat::forge(new_data, object$blueprint)
  data <- resolve_data(processed$predictors, y = data.frame(rep(1, nrow(x))))
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

plot.tabnet_explain <- function(x, ...) {

  df <- rbind(x[[1]], do.call(rbind, x[[2]]))
  df$.type <- rep(c("Aggregated masks", sprintf("Step %d", seq_along(x[[2]]))), each = nrow(x[[1]]))
  df$.rowid <- rep(seq_len(nrow(x[[1]])), times = 1 + length(x[[2]]))

  df <- tidyr::pivot_longer(df, c(-.type, -.rowid), names_to = "variable", values_to = "mask")
  df <- df %>%
    dplyr::group_by(.type) %>%
    dplyr::mutate(mask = trim_quantile(mask)) %>%
    dplyr::ungroup()
 # df <- df %>% dplyr::filter(.type == "Aggregated masks")

  ggplot2::ggplot(df, ggplot2::aes(x = .rowid, y = variable, fill = mask)) +
    ggplot2::geom_tile() +
    ggplot2::facet_wrap(~.type) +
    ggplot2::scale_fill_viridis_c()
}

trim_quantile <- function(x) {
  q99 <- quantile(x, probs = 0.99)
  ifelse(x > q99, q99, x)
}

