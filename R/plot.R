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
    dplyr::mutate(mean_loss = map_dbl(loss, mean),
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
