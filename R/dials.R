check_dials <- function() {
  if (!requireNamespace("dials", quietly = TRUE))
    stop("Package \"dials\" needed for this function to work. Please install it.", call. = FALSE)
}

check_cli <- function() {
  if (!requireNamespace("cli", quietly = TRUE))
    stop("Package \"cli\" needed for this function to work. Please install it.", call. = FALSE)
}



#' Parameters for the tabnet model
#'
#' @param range the default range for the parameter value
#' @param trans whether to apply a transformation to the parameter
#' @param values possible values for factor parameters
#'
#' These functions are used with `tune` grid functions to generate
#' candidates.
#'
#' @rdname tabnet_params
#' @return A `dials` parameter to be used when tuning TabNet models.
#' @export
#' @examplesIf (require("dials") && require("parsnip") && torch::torch_is_installed())
#'   model <- tabnet(attention_width = tune(), feature_reusage = tune(),
#'     momentum = tune(), penalty = tune(), rate_step_size = tune()) %>%
#'     parsnip::set_mode("regression") %>%
#'     parsnip::set_engine("torch")
#'
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
momentum <- function(range = c(0.01, 0.4), trans = NULL) {
  check_dials()
  dials::new_quant_param(
    type = "double",
    range = range,
    inclusive = c(TRUE, TRUE),
    trans = trans,
    label = c(momentum = "Momentum for batch normalization"),
    finalize = NULL
  )
}


#' @rdname tabnet_params
#' @export
mask_type <- function(values = c("sparsemax", "entmax")) {
  check_dials()
  dials::new_qual_param(
    type = "character",
    values = values,
    label = c(mask_type = "Final layer of feature selector, either sparsemax or entmax"),
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

#' Non-tunable parameters for the tabnet model
#'
#' @param range unused
#' @param trans unused
#' @rdname tabnet_non_tunable
#' @export
cat_emb_dim <- function(range = NULL, trans = NULL) {
  check_cli()
  cli::cli_abort("{.var cat_emb_dim} cannot be used as a {.fun tune} parameter yet.")
}

#' @rdname tabnet_non_tunable
#' @export
checkpoint_epochs <- cat_emb_dim

#' @rdname tabnet_non_tunable
#' @export
drop_last <- cat_emb_dim

#' @rdname tabnet_non_tunable
#' @export
encoder_activation <- cat_emb_dim

#' @rdname tabnet_non_tunable
#' @export
lr_scheduler <- cat_emb_dim

#' @rdname tabnet_non_tunable
#' @export
mlp_activation <- cat_emb_dim

#' @rdname tabnet_non_tunable
#' @export
mlp_hidden_multiplier <- cat_emb_dim

#' @rdname tabnet_non_tunable
#' @export
num_independent_decoder <- cat_emb_dim

#' @rdname tabnet_non_tunable
#' @export
num_shared_decoder <- cat_emb_dim

#' @rdname tabnet_non_tunable
#' @export
optimizer <- cat_emb_dim

#' @rdname tabnet_non_tunable
#' @export
penalty <- cat_emb_dim

#' @rdname tabnet_non_tunable
#' @export
verbose <- cat_emb_dim

#' @rdname tabnet_non_tunable
#' @export
virtual_batch_size <- cat_emb_dim
