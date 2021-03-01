#' Plot tabnet_fit loss along epochs
#'
#' @param object A `tabnet_fit` or `tabnet_pretrain` object as a result of
#' [tabnet_fit()] or [tabnet_pretrain()].
#' @param ...  not used.
#' @return A `ggplot2` object.
#' @details
#'  Plot the training loss along epochs, and validation loss along epochs if any.
#'  A diamond symbol is added on epochs where model snapshot is available.
#'
#' @examples
#' \donttest{
#' data("ames", package = "modeldata")
#' ames_fit <- tabnet_fit(Sale_Price ~. , data=ames, epoch=15)
#'
#' # Plot the model loss over epochs
#' autoplot(ames_fit)
#' }
#' @export
autoplot.tabnet_fit <- function(object, ...) {
  epoch_checkpointed_seq <- seq_along(object$fit$checkpoints) * object$fit$config$checkpoint_epochs
  collect_metrics <- tibble::enframe(object$fit$metrics,name = "epoch") %>%
    tidyr::unnest_longer(value,indices_to = "dataset") %>%
    tidyr::unnest_wider(value) %>%
    dplyr::mutate(mean_loss = purrr::map_dbl(loss, mean),
           has_checkpoint = epoch %in% epoch_checkpointed_seq)
  checkpoints <- collect_metrics %>% dplyr::filter(has_checkpoint, dataset=="train")
  p <- ggplot(collect_metrics, aes(x=epoch, y=mean_loss, color=dataset)) +
    geom_point(data = checkpoints, aes(x=epoch, y=mean_loss, color=dataset), shape=5) +
    geom_line() +
    scale_y_log10()
  p
  }

#' @export
autoplot.tabnet_pretrain <- autoplot.tabnet_fit

#' Plot tabnet_explain interpretation
#'
#' @param object A `tabnet_explain` object as a result of [tabnet_explain()].
#' @param type a character value. Either `"mask_agg"` the default, for per predictor
#' mask importance aggregate,or `"steps"` for a plot at each mask step.
#' @param ...  not used.
#' @return A `ggplot2` object.
#' @details
#'  Plot the tabnet_explain object mask importance per variable along the predicted dataset.
#'  `type="mask_agg"` output a single heatmap of mask aggregated values,
#'  `type="mask_agg"` provides a facet plot of each mask along the model.
#'
#' @examples
#' \donttest{
#' data("ames", package = "modeldata")
#' ames_fit <- tabnet_fit(Sale_Price ~. , data=ames, epoch=15)
#' ames_explain <- tabnet_explain(ames_fit, ames)
#' # Plot the model mask interpretation heatmap
#' autoplot(ames_explain)
#' }

#'  @export
autoplot.tabnet_explain <- function(object, type = c("mask_agg", "steps"), ...) {
  type <- match.arg(type)

  if (type == "steps") {
    .data <- object$masks %>%
      purrr::imap_dfr(~mutate(
        .x,
        step = sprintf("Step %d", .y),
        rowname = row_number()
      )) %>%
      tidyr::pivot_longer(-c(rowname, step), names_to = "variable", values_to = "mask_agg") %>%
      dplyr::group_by(step) %>%
      dplyr::mutate(mask_agg = min(mask_agg , q995(mask_agg))) %>%
      dplyr::ungroup()
  } else {

  .data <- object$M_explain %>%
    dplyr::mutate(rowname = row_number()) %>%
    tidyr::pivot_longer(-rowname, names_to = "variable", values_to = "mask_agg") %>%
    dplyr::mutate(mask_agg = min(mask_agg , q995(mask_agg)),
                  step = "mask_aggregate")
  }

  p <- ggplot(.data, aes(x = rowname, y = variable, fill = mask_agg)) +
    geom_tile() +
    scale_fill_viridis_c() +
    facet_wrap(~step)
  p
}

q995 <- function(x) quantile(x, probs = 0.995)
