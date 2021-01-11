#' Tabnet model
#'
#' Fits the [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442) model
#'
#' @param x Depending on the context:
#'
#'   * A __data frame__ of predictors.
#'   * A __matrix__ of predictors.
#'   * A __recipe__ specifying a set of preprocessing steps
#'     created from [recipes::recipe()].
#'
#'  The predictor data should be standardized (e.g. centered or scaled).
#'  The model treats categorical predictors internally thus, you don't need to
#'  make any treatment.
#'
#' @param y When `x` is a __data frame__ or __matrix__, `y` is the outcome
#' specified as:
#'
#'   * A __data frame__ with 1 numeric column.
#'   * A __matrix__ with 1 numeric column.
#'   * A numeric __vector__.
#'
#' @param data When a __recipe__ or __formula__ is used, `data` is specified as:
#'
#'   * A __data frame__ containing both the predictors and the outcome.
#'
#' @param formula A formula specifying the outcome terms on the left-hand side,
#'  and the predictor terms on the right-hand side.
#' @param ... Model hyperparameters. See [tabnet_config()] for a list of
#'  all possible hyperparameters.
#'
#' @section Threading:
#'
#' TabNet uses `torch` as it's backend for computation and `torch` uses all
#' available threads by default.
#'
#' You can control the number of threads used by `torch` with:
#'
#' ```
#' torch::torch_set_num_threads(1)
#' torch::torch_set_num_interop_threads(1)
#' ```
#'
#' @examples
#' if (torch::torch_is_installed()) {
#' data("ames", package = "modeldata")
#' fit <- tabnet_fit(Sale_Price ~ ., data = ames, epochs = 1)
#' }
#'
#' @return A TabNet model object. It can be used for serialization and predictions.
#'
#' @export
tabnet_fit <- function(x, ...) {
  UseMethod("tabnet_fit")
}

#' @export
#' @rdname tabnet_fit
tabnet_fit.default <- function(x, ...) {
  stop(
    "`tabnet_fit()` is not defined for a '", class(x)[1], "'.",
    call. = FALSE
  )
}

#' @export
#' @rdname tabnet_fit
tabnet_fit.data.frame <- function(x, y, ...) {
  processed <- hardhat::mold(x, y)
  config <- do.call(tabnet_config, list(...))
  tabnet_bridge(processed, config = config)
}

#' @export
#' @rdname tabnet_fit
tabnet_fit.formula <- function(formula, data, ...) {
  processed <- hardhat::mold(
    formula, data,
    blueprint = hardhat::default_formula_blueprint(
      indicators = "none",
      intercept = FALSE
    )
  )
  config <- do.call(tabnet_config, list(...))
  tabnet_bridge(processed, config = config)
}

#' @export
#' @rdname tabnet_fit
tabnet_fit.recipe <- function(x, data, ...) {
  processed <- hardhat::mold(x, data)
  config <- do.call(tabnet_config, list(...))
  tabnet_bridge(processed, config = config)
}

new_tabnet_fit <- function(fit, blueprint) {

  serialized_net <- model_to_raw(fit$network)

  hardhat::new_model(
    fit = fit,
    serialized_net = serialized_net,
    blueprint = blueprint,
    class = "tabnet_fit"
  )
}

tabnet_bridge <- function(processed, config = tabnet_config()) {
  predictors <- processed$predictors
  outcomes <- processed$outcomes
  fit <- tabnet_impl(predictors, outcomes, config = config)
  new_tabnet_fit(fit, blueprint = processed$blueprint)
}

#' @importFrom stats predict
#' @export
predict.tabnet_fit <- function(object, new_data, type = NULL, ..., epoch = NULL) {
  # Enforces column order, type, column names, etc
  processed <- hardhat::forge(new_data, object$blueprint)
  out <- predict_tabnet_bridge(type, object, processed$predictors, epoch)
  hardhat::validate_prediction_size(out, new_data)
  out
}

check_type <- function(model, type) {

  outcome_ptype <- model$blueprint$ptypes$outcomes[[1]]

  if (is.null(type)) {
    if (is.factor(outcome_ptype))
      type <- "class"
    else if (is.numeric(outcome_ptype))
      type <- "numeric"
    else
      rlang::abort(glue::glue("Unknown outcome type '{class(outcome_ptype)}'"))
  }

  type <- rlang::arg_match(type, c("numeric", "prob", "class"))

  if (is.factor(outcome_ptype)) {
    if (!type %in% c("prob", "class"))
      rlang::abort(glue::glue("Outcome is factor and the prediction type is '{type}'."))
  } else if (is.numeric(outcome_ptype)) {
    if (type != "numeric")
      rlang::abort(glue::glue("Outcome is numeric and the prediction type is '{type}'."))
  }

  type
}



predict_tabnet_bridge <- function(type, object, predictors, epoch) {

  type <- check_type(object, type)

  if (!is.null(epoch)) {

    if (epoch > (length(object$fit$checkpoints) * object$fit$config$checkpoint_epoch))
      rlang::abort(paste0("The model was trained for less than ", epoch, " epochs"))

    # find closest checkpoint for that epoch
    ind <- epoch %/% object$fit$config$checkpoint_epoch

    object$fit$network <- reload_model(object$fit$checkpoints[[ind]])
  }

  if (check_net_is_empty_ptr(object)) {
    m <- reload_model(object$serialized_net)
    # this modifies 'object' in-place so subsequent predicts won't
    # need to reload.
    object$fit$network$load_state_dict(m$state_dict())
  }

  switch(
    type,
    numeric = predict_impl_numeric(object, predictors),
    prob    = predict_impl_prob(object, predictors),
    class   = predict_impl_class(object, predictors)
  )
}

model_to_raw <- function(model) {
  con <- rawConnection(raw(), open = "wr")
  torch::torch_save(model, con)
  on.exit({close(con)}, add = TRUE)
  r <- rawConnectionValue(con)
  r
}

check_net_is_empty_ptr <- function(object) {
  is_null_external_pointer(object$fit$network$.check$ptr)
}

# https://stackoverflow.com/a/27350487/3297472
is_null_external_pointer <- function(pointer) {
  a <- attributes(pointer)
  attributes(pointer) <- NULL
  out <- identical(pointer, methods::new("externalptr"))
  attributes(pointer) <- a
  out
}

reload_model <- function(object) {
  con <- rawConnection(object)
  on.exit({close(con)}, add = TRUE)
  module <- torch::torch_load(con)
  module
}

#' @export
print.tabnet_fit <- function(x, ...) {
  print(x$fit$network)
  invisible(x)
}
