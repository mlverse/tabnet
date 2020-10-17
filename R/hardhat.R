tabnet_fit <- function(x, ...) {
  UseMethod("tabnet")
}

tabnet_fit.default <- function(x, ...) {
  stop(
    "`tabnet()` is not defined for a '", class(x)[1], "'.",
    call. = FALSE
  )
}

tabnet_fit.data.frame <- function(x, y, valid_data, ...) {
  processed <- hardhat::mold(x, y)
  tabnet_bridge(processed, valid_data, config = list(...))
}

tabnet_fit.formula <- function(formula, data, ...) {
  processed <- mold(formula, data)
  tabnet_bridge(processed)
}

tabnet_fit.recipe <- function(x, data, ...) {
  processed <- mold(x, data)
  tabnet_bridge(processed)
}

new_tabnet <- function(fit, blueprint) {
  fit$blueprint <- blueprint
  fit$class <- "tabnet"
  do.call(hardhat::new_model, fit)
}

tabnet_bridge <- function(processed, valid_data = NULL, config = tabnet_config()) {

  predictors <- processed$predictors
  outcomes <- processed$outcomes

  fit <- tabnet_impl(predictor, outcomes, valid_data, config)

  new_tabnet(fit, blueprint = processed$blueprint)
}
