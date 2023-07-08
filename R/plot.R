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
#' @examplesIf torch::torch_is_installed()
#' library(ggplot2)
#' data("attrition", package = "modeldata")
#' attrition_fit <- tabnet_fit(Attrition ~. , data=attrition, valid_split=0.2, epoch=11)
#'
#' # Plot the model loss over epochs
#' autoplot(attrition_fit)
#'
#' @importFrom rlang .data
#' @importFrom dplyr filter mutate select select_if
#' @importFrom ggplot2 aes element_text geom_line geom_point ggplot guide_legend guides labs scale_y_log10 theme
#' @importFrom purrr map_dbl
#' @importFrom tibble enframe
#' @importFrom tidyr drop_na pivot_longer pivot_wider unnest_longer unnest_wider
#'
autoplot.tabnet_fit <- function(object, ...) {

  epoch_checkpointed_seq <- seq_along(object$fit$checkpoints) * object$fit$config$checkpoint_epochs

  collect_metrics <- tibble::enframe(object$fit$metrics,name = "epoch") %>%
    unnest_longer(value,indices_to = "dataset") %>%
    unnest_wider(value) %>%
    # drop entries from pretrain that have missing `dataset`
    drop_na(dataset) %>%
    pivot_wider(values_from = loss, names_from = dataset) %>%
    # remove the valid col if all NAs to prevent ggplot warnings
    select_if(function(x) {!all(is.na(x))} ) %>%
    pivot_longer(cols = !epoch, names_to = "dataset", values_to = "loss") %>%
    # add checkpoints
    mutate(mean_loss = map_dbl(loss, mean),
           has_checkpoint = epoch %in% (epoch_checkpointed_seq + min(epoch, na.rm=TRUE) - 1)) %>%
    select(-loss)

  checkpoints <- collect_metrics %>%
    filter(has_checkpoint, dataset=="train") %>%
    mutate(size=2)
  p <- ggplot(collect_metrics, aes(x=epoch, y=mean_loss, color=dataset)) +
    geom_line() +
    geom_point(data = checkpoints, aes(x=epoch, y=mean_loss, color=dataset, size = .data$size ) ) +
    scale_y_log10() +
    guides(colour = guide_legend("Dataset", order=1, override.aes = list(size=1.5, shape=" ")),
           size= guide_legend("has checkpoint", order=2, override.aes = list(size=3, color="#F8766D"), label.theme = element_text(colour = "#FFFFFF"))) +
    theme(legend.position = "bottom") +
    labs(y="Mean loss (log scale)")
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
#' @examplesIf torch::torch_is_installed()
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
#' ids <- sample(nrow(ames), 256)
#' x <- ames[ids,-which(names(ames) %in% c("Sale_Price", "Pool_Area"))]
#' y <- ames[ids, c("Sale_Price", "Pool_Area")]
#' ames_fit <- tabnet_fit(x, y, epochs = 5, verbose=TRUE)
#' ames_explain <- tabnet_explain(ames_fit, x)
#' autoplot(ames_explain, quantile = 0.99)
#' @importFrom dplyr group_by mutate row_number ungroup
#' @importFrom ggplot2 aes facet_wrap geom_tile ggplot scale_fill_viridis_c theme_minimal
#' @importFrom purrr imap_dfr
#' @importFrom tidyr pivot_longer
autoplot.tabnet_explain <- function(object, type = c("mask_agg", "steps"), quantile = 1, ...) {
  type <- match.arg(type)

  if (type == "steps") {
    .data <- object$masks %>%
      imap_dfr(~mutate(
        .x,
        step = sprintf("Step %d", .y),
        rowname = row_number()
      )) %>%
      pivot_longer(-c(rowname, step), names_to = "variable", values_to = "mask_agg") %>%
      group_by(step) %>%
      mutate(mask_agg = quantile_clip(mask_agg, probs=quantile)) %>%
      ungroup()
  } else {

  .data <- object$M_explain %>%
    mutate(rowname = row_number()) %>%
    pivot_longer(-rowname, names_to = "variable", values_to = "mask_agg") %>%
    mutate(mask_agg = quantile_clip(mask_agg, probs=quantile),
                  step = "mask_aggregate")
  }

  p <- ggplot(.data, aes(x = rowname, y = variable, fill = mask_agg)) +
    geom_tile() +
    scale_fill_viridis_c() +
    facet_wrap(~step) +
    theme_minimal()
  p
}

#' @importFrom purrr map_dbl
#' @importFrom stats quantile
quantile_clip <- function(x, probs) {
  quantile <- quantile(x, probs = probs)
  map_dbl(x, ~min(.x, quantile))
}
