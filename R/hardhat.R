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
#' and the predictor terms on the right-hand side.
#' @param config The config parameters. See [tabnet_config()].
#'
#'
#' @export
tabnet_fit <- function(x, ...) {
  UseMethod("tabnet_fit")
}

#' @export
tabnet_fit.default <- function(x, ...) {
  stop(
    "`tabnet()` is not defined for a '", class(x)[1], "'.",
    call. = FALSE
  )
}

#' @export
tabnet_fit.data.frame <- function(x, y, ..., config) {
  processed <- hardhat::mold(x, y)
  tabnet_bridge(processed, config = config)
}

#' @export
tabnet_fit.formula <- function(formula, data, ..., config) {
  processed <- mold(
    formula, data,
    blueprint = hardhat::default_formula_blueprint(
      indicators = FALSE,
      intercept = FALSE
    )
  )
  tabnet_bridge(processed, config = config)
}

#' @export
tabnet_fit.recipe <- function(x, data, ..., config) {
  processed <- mold(x, data)
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
  new_tabnet(fit, blueprint = processed$blueprint)
}
