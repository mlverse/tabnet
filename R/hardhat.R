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
#' @examples
#' if (torch::torch_is_installed()) {
#' data("ames", package = "modeldata")
#' fit <- tabnet_fit(Sale_Price ~ ., data = ames, epochs = 1)
#' }
#' @export
tabnet_fit <- function(x, ...) {
  UseMethod("tabnet_fit")
}

#' @export
tabnet_fit.default <- function(x, ...) {
  stop(
    "`tabnet_fit()` is not defined for a '", class(x)[1], "'.",
    call. = FALSE
  )
}

#' @export
tabnet_fit.data.frame <- function(x, y, ...) {
  processed <- hardhat::mold(x, y)
  config <- do.call(tabnet_config, list(...))
  tabnet_bridge(processed, config = config)
}

#' @export
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
tabnet_fit.recipe <- function(x, data, ...) {
  processed <- mold(x, data)
  config <- do.call(tabnet_config, list(...))
  tabnet_bridge(processed, config = config)
}

new_tabnet_fit <- function(fit, blueprint) {
  hardhat::new_model(
    fit = fit,
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

#' @export
predict.tabnet_fit <- function(object, new_data, type = NULL, ...) {
  # Enforces column order, type, column names, etc
  processed <- hardhat::forge(new_data, object$blueprint)
  out <- predict_tabnet_bridge(type, object, processed$predictors)
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

predict_tabnet_bridge <- function(type, object, predictors) {

  type <- check_type(object, type)

  switch(
    type,
    numeric = predict_impl_numeric(object, predictors),
    prob    = predict_impl_prob(object, predictors),
    class   = predict_impl_class(object, predictors)
  )
}
