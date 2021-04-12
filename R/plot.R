#' Plot tabnet_fit model loss along epochs
#'
#' @param object A `tabnet_fit` or `tabnet_pretrain` object as a result of
#' [tabnet_fit()] or [tabnet_pretrain()].
#' @param ...  not used.
#' @return A `ggplot` object.
#' @details
#'  Plot the training loss along epochs, and validation loss along epochs if any.
#'  A dot is added on epochs where model snapshot is available, helping
#'  the choice of `from_epoch` value for later model training resume.
#'
#' @examples
#' \donttest{
#' library(ggplot2)
#' data("attrition", package = "modeldata")
#' attrition_fit <- tabnet_fit(Attrition ~. , data=attrition, valid_split=0.2, epoch=15)
#'
#' # Plot the model loss over epochs
#' autoplot(attrition_fit)
#' }
#'
autoplot.tabnet_fit <- function(object, ...) {

  epoch_checkpointed_seq <- seq_along(object$fit$checkpoints) * object$fit$config$checkpoint_epochs

  collect_metrics <- tibble::enframe(object$fit$metrics,name = "epoch") %>%
    tidyr::unnest_longer(value,indices_to = "dataset") %>%
    tidyr::unnest_wider(value) %>%
    # remove the valid col if all NAs to prevent ggplot warnings
    tidyr::pivot_wider(values_from = loss, names_from = dataset) %>%
    dplyr::select_if(~!all(is.na(.x))) %>%
    tidyr::pivot_longer(cols = !epoch, names_to = "dataset", values_to = "loss") %>%
    # add checkpoints
    dplyr::mutate(mean_loss = purrr::map_dbl(loss, mean),
           has_checkpoint = epoch %in% epoch_checkpointed_seq) %>%
    dplyr::select(-loss)

  checkpoints <- collect_metrics %>% dplyr::filter(has_checkpoint, dataset=="train")
  p <- ggplot2::ggplot(collect_metrics, ggplot2::aes(x=epoch, y=mean_loss, color=dataset)) +
    ggplot2::geom_point(data = checkpoints, ggplot2::aes(x=epoch, y=mean_loss, color=dataset), size = 2 ) +
    ggplot2::geom_line() +
    ggplot2::scale_y_log10()
  p
  }

#' @rdname autoplot.tabnet_fit
autoplot.tabnet_pretrain <- autoplot.tabnet_fit

#' Plot tabnet_explain mask importance heatmap
#'
#' @param object A `tabnet_explain` object as a result of [tabnet_explain()].
#' @param type a character value. Either `"mask_agg"` the default, for a single
#'  heatmap of aggregated mask importance per predictor along the dataset,
#'   or `"steps"` for one heatmap at each mask step.
#' @param quantile numerical value between 0 and 1. Provides quantile clipping of the
#'  mask values
#' @param ...  not used.
#' @return A `ggplot` object.
#' @details
#'  Plot the tabnet_explain object mask importance per variable along the predicted dataset.
#'  `type="mask_agg"` output a single heatmap of mask aggregated values,
#'  `type="steps"` provides a plot faceted along the `n_steps` mask present in the model.
#'  `quantile=.995` may be used for strong outlier clipping, in order to better highlight
#'  low values. `quantile=1`, the default, do not clip any values.
#'
#' @examples
#' \donttest{
#' library(ggplot2)
#' data("attrition", package = "modeldata")
#' attrition_fit <- tabnet_fit(Attrition ~. , data=attrition, epoch=15)
#' attrition_explain <- tabnet_explain(attrition_fit, attrition)
#' # Plot the model aggregated mask interpretation heatmap
#' autoplot(attrition_explain)
#' }
#'
#'
autoplot.tabnet_explain <- function(object, type = c("mask_agg", "steps"), quantile = 1, ...) {
  type <- match.arg(type)

  if (type == "steps") {
    .data <- object$masks %>%
      purrr::imap_dfr(~dplyr::mutate(
        .x,
        step = sprintf("Step %d", .y),
        rowname = dplyr::row_number()
      )) %>%
      tidyr::pivot_longer(-c(rowname, step), names_to = "variable", values_to = "mask_agg") %>%
      dplyr::group_by(step) %>%
      dplyr::mutate(mask_agg = quantile_clip(mask_agg, probs=quantile)) %>%
      dplyr::ungroup()
  } else {

  .data <- object$M_explain %>%
    dplyr::mutate(rowname = dplyr::row_number()) %>%
    tidyr::pivot_longer(-rowname, names_to = "variable", values_to = "mask_agg") %>%
    dplyr::mutate(mask_agg = quantile_clip(mask_agg, probs=quantile),
                  step = "mask_aggregate")
  }

  p <- ggplot2::ggplot(.data, ggplot2::aes(x = rowname, y = variable, fill = mask_agg)) +
    ggplot2::geom_tile() +
    ggplot2::scale_fill_viridis_c() +
    ggplot2::facet_wrap(~step) +
    ggplot2::theme_minimal()
  p
}

quantile_clip <- function(x, probs) {
  quantile <- quantile(x, probs = probs)
  purrr::map_dbl(x, ~min(.x, quantile))
}
