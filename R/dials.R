#' Parameters for the tabnet model
#'
#' @param range the default range for the parameter value
#' @param trans wether to apply a transformation to the parameter
#'
#'
#' @rdname tabnet_params
#' @export
decision_width <- function(range = c(8L, 64L), trans = NULL) {
  dials::new_quant_param(
    type = "integer",
    range = range,
    inclusive = c(TRUE, TRUE),
    trans = trans,
    label = c(min_n = "Width of the decision prediction layer"),
    finalize = NULL
  )
}

#' @rdname tabnet_params
#' @export
attention_width <- function(range = c(8L, 64L), trans = NULL) {
  dials::new_quant_param(
    type = "integer",
    range = range,
    inclusive = c(TRUE, TRUE),
    trans = trans,
    label = c(min_n = "Width of the attention embedding for each mask"),
    finalize = NULL
  )
}

#' @rdname tabnet_params
#' @export
num_steps <- function(range = c(3L, 10L), trans = NULL) {
  dials::new_quant_param(
    type = "integer",
    range = range,
    inclusive = c(TRUE, TRUE),
    trans = trans,
    label = c(min_n = "Number of steps in the architecture"),
    finalize = NULL
  )
}

#' @rdname tabnet_params
#' @export
feature_reusage <- function(range = c(1, 2), trans = NULL) {
  dials::new_quant_param(
    type = "double",
    range = range,
    inclusive = c(TRUE, TRUE),
    trans = trans,
    label = c(min_n = "Coefficient for feature reusage in the masks"),
    finalize = NULL
  )
}
