check_dials <- function() {
  if (!requireNamespace("dials", quietly = TRUE))
    rlang::abort("Package \"dials\" needed for this function to work. Please install it.")
}



#' Parameters for the tabnet model
#'
#' @param range the default range for the parameter value
#' @param trans wether to apply a transformation to the parameter
#'
#' These functions are used with `tune` grid functions to generate
#' candidates.
#'
#' @rdname tabnet_params
#' @return A `dials` parameter to be used when tuning TabNet models.
#' @export
decision_width <- function(range = c(8L, 64L), trans = NULL) {
  check_dials()
  dials::new_quant_param(
    type = "integer",
    range = range,
    inclusive = c(TRUE, TRUE),
    trans = trans,
    label = c(decision_width = "Width of the decision prediction layer"),
    finalize = NULL
  )
}

#' @rdname tabnet_params
#' @export
attention_width <- function(range = c(8L, 64L), trans = NULL) {
  check_dials()
  dials::new_quant_param(
    type = "integer",
    range = range,
    inclusive = c(TRUE, TRUE),
    trans = trans,
    label = c(attention_width = "Width of the attention embedding for each mask"),
    finalize = NULL
  )
}

#' @rdname tabnet_params
#' @export
num_steps <- function(range = c(3L, 10L), trans = NULL) {
  check_dials()
  dials::new_quant_param(
    type = "integer",
    range = range,
    inclusive = c(TRUE, TRUE),
    trans = trans,
    label = c(num_steps = "Number of steps in the architecture"),
    finalize = NULL
  )
}

#' @rdname tabnet_params
#' @export
feature_reusage <- function(range = c(1, 2), trans = NULL) {
  check_dials()
  dials::new_quant_param(
    type = "double",
    range = range,
    inclusive = c(TRUE, TRUE),
    trans = trans,
    label = c(feature_reusage = "Coefficient for feature reusage in the masks"),
    finalize = NULL
  )
}

#' @rdname tabnet_params
#' @export
num_independent <- function(range = c(1L, 5L), trans = NULL) {
  check_dials()
  dials::new_quant_param(
    type = "integer",
    range = range,
    inclusive = c(TRUE, TRUE),
    trans = trans,
    label = c(num_independent = "Number of independent Gated Linear Units layers at each step"),
    finalize = NULL
  )
}

#' @rdname tabnet_params
#' @export
num_shared <- function(range = c(1L, 5L), trans = NULL) {
  check_dials()
  dials::new_quant_param(
    type = "integer",
    range = range,
    inclusive = c(TRUE, TRUE),
    trans = trans,
    label = c(num_shared = "Number of shared Gated Linear Units at each step"),
    finalize = NULL
  )
}
