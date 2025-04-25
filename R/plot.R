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
#' @examplesIf (torch::torch_is_installed() && require("modeldata"))
#'  \dontrun{
#' library(ggplot2)
#' data("attrition", package = "modeldata")
#' attrition_fit <- tabnet_fit(Attrition ~. , data=attrition, valid_split=0.2, epoch=11)
#'
#' # Plot the model loss over epochs
#' autoplot(attrition_fit)
#'}
#' @importFrom rlang .data
#'
autoplot.tabnet_fit <- function(object, ...) {

  collect_metrics <- tibble::enframe(object$fit$metrics, name = "epoch") %>%
    tidyr::unnest_wider(value) %>%
    dplyr::mutate_if(is.list, ~purrr::map_dbl(.x, mean)) %>%
    dplyr::select_if(function(x) {!all(is.na(x))} ) %>%
    tidyr::pivot_longer(cols = !dplyr::matches("epoch|checkpoint"),
                        names_to = "dataset", values_to = "loss")


  p <- ggplot2::ggplot(collect_metrics, ggplot2::aes(x = epoch, y = loss, color = dataset)) +
    ggplot2::geom_line() +
    ggplot2::scale_y_log10() +
    ggplot2::guides(colour = ggplot2::guide_legend("Dataset", order=1, override.aes = list(size = 1.7, shape = " ")),
                    size = ggplot2::guide_legend("has checkpoint", order = 2, override.aes = list(size = 3, color = "#F8766D"),
                                                 label.theme = ggplot2::element_text(colour = "#FFFFFF"))) +
    ggplot2::theme(legend.position = "bottom") +
    ggplot2::labs(y="Mean loss (log scale)")

  if ("checkpoint" %in% names(collect_metrics)) {
    checkpoints <- collect_metrics %>%
      dplyr::filter(checkpoint == TRUE, dataset == "train") %>%
      dplyr::select(-checkpoint) %>%
      dplyr::mutate(size = 2)
    p +
      ggplot2::geom_point(data = checkpoints, ggplot2::aes(x = epoch, y = loss, color = dataset, size = .data$size ))
  } else {
    p
  }
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
#'  Plot the `tabnet_explain` object mask importance per variable along the predicted dataset.
#'  `type="mask_agg"` output a single heatmap of mask aggregated values,
#'  `type="steps"` provides a plot faceted along the `n_steps` mask present in the model.
#'  `quantile=.995` may be used for strong outlier clipping, in order to better highlight
#'  low values. `quantile=1`, the default, do not clip any values.
#'
#' @examplesIf (torch::torch_is_installed() && require("modeldata"))
#'  \dontrun{
#' library(ggplot2)
#' data("attrition", package = "modeldata")
#'
#' ## Single-outcome binary classification of `Attrition` in `attrition` dataset
#' attrition_fit <- tabnet_fit(Attrition ~. , data=attrition, epoch=11)
#' attrition_explain <- tabnet_explain(attrition_fit, attrition)
#' # Plot the model aggregated mask interpretation heatmap
#' autoplot(attrition_explain)
#'
#' ## Multi-outcome regression on `Sale_Price` and `Pool_Area` in `ames` dataset,
#' data("ames", package = "modeldata")
#' x <- ames[,-which(names(ames) %in% c("Sale_Price", "Pool_Area"))]
#' y <- ames[, c("Sale_Price", "Pool_Area")]
#' ames_fit <- tabnet_fit(x, y, epochs = 1, verbose=TRUE)
#' ames_explain <- tabnet_explain(ames_fit, x)
#' autoplot(ames_explain, quantile = 0.99)
#' }
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
