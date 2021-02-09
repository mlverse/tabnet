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
#' @param tabnet_model A previously fitted TabNet model object to continue the fitting on.
#'  if `NULL` (the default) a brand new model is initialized.
#' @param from_epoch When a `tabnet_model` is provided, restore the network weights from a specific epoch.
#'  Default is last available checkpoint for restored model, or last epoch for in-memory model.
#' @param ... Model hyperparameters. See [tabnet_config()] for a list of
#'  all possible hyperparameters.
#'
#' @section Fitting a pre-trained model:
#'
#' When providing a parent `tabnet_model` parameter, the model fitting resumes from that model weights
#' at the following epoch:
#'    * last fitted epoch for a model already in torch context
#'    * Last model checkpoint epoch for a model loaded from file
#'    * the epoch related to a checkpoint matching or preceding the `from_epoch` value if provided
#' The model fitting metrics append on top of the parent metrics in the returned TabNet model.
#'
#' @section Threading:
#'
#' TabNet uses `torch` as its backend for computation and `torch` uses all
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
#' @return A TabNet model object. It can be used for serialization, predictions, or further fitting.
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
tabnet_fit.data.frame <- function(x, y, tabnet_model = NULL, ..., from_epoch = NULL) {
  processed <- hardhat::mold(x, y)
  config <- do.call(tabnet_config, list(...))
  tabnet_bridge(processed, config = config, tabnet_model, from_epoch, task="supervised")
}

#' @export
#' @rdname tabnet_fit
tabnet_fit.formula <- function(formula, data, tabnet_model = NULL, ..., from_epoch = NULL) {
  processed <- hardhat::mold(
    formula, data,
    blueprint = hardhat::default_formula_blueprint(
      indicators = "none",
      intercept = FALSE
    )
  )
  config <- do.call(tabnet_config, list(...))
  tabnet_bridge(processed, config = config, tabnet_model, from_epoch, task="supervised")
}

#' @export
#' @rdname tabnet_fit
tabnet_fit.recipe <- function(x, data, tabnet_model = NULL, ..., from_epoch = NULL) {
  processed <- hardhat::mold(x, data)
  config <- do.call(tabnet_config, list(...))
  tabnet_bridge(processed, config = config, tabnet_model, from_epoch, task="supervised")
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

#' Tabnet model
#'
#' Pretrain the [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442) model
#' on the predictor data exclusively (unsupervised training).
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
#' @param y (optional) When `x` is a __data frame__ or __matrix__, `y` is the outcome
#' @param data When a __recipe__ or __formula__ is used, `data` is specified as:
#'
#'   * A __data frame__ containing both the predictors and the outcome.
#'
#' @param formula A formula specifying the outcome terms on the left-hand side,
#'  and the predictor terms on the right-hand side.
#' @param tabnet_model A pretrained TabNet model object to continue the fitting on.
#'  if `NULL` (the default) a brand new model is initialized.
#' @param from_epoch When a `tabnet_model` is provided, restore the network weights from a specific epoch.
#'  Default is last available checkpoint for restored model, or last epoch for in-memory model.
#' @param ... Model hyperparameters. See [tabnet_config()] for a list of
#'  all possible hyperparameters.
#'
#' @section outcome:
#'
#' Outcome value are accepted here only for consistent syntax with `tabnet_fit`, but
#' by design the outcome, if present, is ignored during pre-training.
#'
#' @section pre-training from a previous model:
#'
#' When providing a parent `tabnet_model` parameter, the model pretraining resumes from that model weights
#' at the following epoch:
#'    * last pretrained epoch for a model already in torch context
#'    * Last model checkpoint epoch for a model loaded from file
#'    * the epoch related to a checkpoint matching or preceding the `from_epoch` value if provided
#' The model pretraining metrics append on top of the parent metrics in the returned TabNet model.
#'
#' @section Threading:
#'
#' TabNet uses `torch` as its backend for computation and `torch` uses all
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
#' pretrained <- tabnet_pretrain(Sale_Price ~ ., data = ames, epochs = 1)
#' }
#'
#' @return A TabNet model object. It can be used for serialization, predictions, or further fitting.
#'
#' @export
tabnet_pretrain <- function(x, ...) {
  UseMethod("tabnet_pretrain")
}

#' @export
#' @rdname tabnet_pretrain
tabnet_pretrain.default <- function(x, ...) {
  stop(
    "`tabnet_pretrain()` is not defined for a '", class(x)[1], "'.",
    call. = FALSE
  )
}

#' @export
#' @rdname tabnet_pretrain
tabnet_pretrain.data.frame <- function(x, y, tabnet_model = NULL, ..., from_epoch = NULL) {
  processed <- hardhat::mold(x, y)
  config <- do.call(tabnet_config, list(...))
  tabnet_bridge(processed, config = config, tabnet_model, from_epoch, task="unsupervised")
}

#' @export
#' @rdname tabnet_pretrain
tabnet_pretrain.formula <- function(formula, data, tabnet_model = NULL, ..., from_epoch = NULL) {
  processed <- hardhat::mold(
    formula, data,
    blueprint = hardhat::default_formula_blueprint(
      indicators = "none",
      intercept = FALSE
    )
  )
  config <- do.call(tabnet_config, list(...))
  tabnet_bridge(processed, config = config, tabnet_model, from_epoch, task="unsupervised")
}

#' @export
#' @rdname tabnet_pretrain
tabnet_pretrain.recipe <- function(x, data, tabnet_model = NULL, ..., from_epoch = NULL) {
  processed <- hardhat::mold(x, data)
  config <- do.call(tabnet_config, list(...))
  tabnet_bridge(processed, config = config, tabnet_model, from_epoch, task="unsupervised")
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

new_tabnet_pretrain <- function(pretrain, blueprint) {

  serialized_net <- model_to_raw(pretrain$network)

  hardhat::new_model(
    fit = pretrain,
    serialized_net = serialized_net,
    blueprint = blueprint,
    class = "tabnet_pretrain"
  )
}

tabnet_bridge <- function(processed, config = tabnet_config(), tabnet_model, from_epoch, task="supervised") {
  predictors <- processed$predictors
  outcomes <- processed$outcomes
  if (!(is.null(tabnet_model) || inherits(tabnet_model, "tabnet_fit") || inherits(tabnet_model, "tabnet_pretrain")))
    rlang::abort(paste0(tabnet_model," is not recognised as a proper TabNet model"))
  if (task == "supervised") {
    if (is.null(tabnet_model)) {
      # new supervised model needs network initialization
      tabnet_model_lst <- tabnet_initialize(predictors, outcomes, config = config)
      tabnet_model <-  new_tabnet_fit(tabnet_model_lst, blueprint = processed$blueprint)
      epoch_shift <- 0L

    } else if (!is.null(from_epoch)) {
      # model must be loaded from checkpoint

      if (from_epoch > (length(tabnet_model$fit$checkpoints) * tabnet_model$fit$config$checkpoint_epoch))
        rlang::abort(paste0("The model was trained for less than ", from_epoch, " epochs"))

      # find closest checkpoint for that epoch
      closest_checkpoint <- from_epoch %/% tabnet_model$fit$config$checkpoint_epoch

      tabnet_model$fit$network <- reload_model(tabnet_model$fit$checkpoints[[closest_checkpoint]])
      epoch_shift <- closest_checkpoint * tabnet_model$fit$config$checkpoint_epoch

    } else if (!check_net_is_empty_ptr(tabnet_model)) {
      # model is available from tabnet_model$serialized_net

      m <- reload_model(tabnet_model$serialized_net)
      # this modifies 'tabnet_model' in-place so subsequent predicts won't
      # need to reload.
      tabnet_model$fit$network$load_state_dict(m$state_dict())
      epoch_shift <- length(tabnet_model$fit$metrics)

    } else if (length(tabnet_model$fit$checkpoints)) {
      # model is loaded from the last available checkpoint

      last_checkpoint <- length(tabnet_model$fit$checkpoints)

      tabnet_model$fit$network <- reload_model(tabnet_model$fit$checkpoints[[last_checkpoint]])
      epoch_shift <- last_checkpoint * tabnet_model$fit$config$checkpoint_epoch

    } else rlang::abort(paste0("No model serialized weight can be found in ", tabnet_model, ", check the model history"))
  }
  if (task == "supervised") {
    fit_lst <- tabnet_train_supervised(tabnet_model, predictors, outcomes, config = config, epoch_shift)
    return(new_tabnet_fit(fit_lst, blueprint = processed$blueprint))
  } else if (task == "unsupervised") {
    pretrain_lst <- tabnet_train_unsupervised( predictors, config = config)
    return(new_tabnet_pretrain(pretrain_lst, blueprint = processed$blueprint))
  }

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
  UseMethod("reload_model")
}
reload_model.default <- function(x, ...) {
  stop(
    "`reload_model()` is not defined for a '", class(x)[1], "'.",
    call. = FALSE
  )
}

reload_model.tabnet_fit <- function(object) {
  con <- rawConnection(object)
  on.exit({close(con)}, add = TRUE)
  module <- torch::torch_load(con)
  module
}

reload_model.tabnet_pretrain <- function(object) {
  con <- rawConnection(object)
  on.exit({close(con)}, add = TRUE)
  module <- torch::torch_load(con)
  # TODO perform ablation of the decoder nn_module and addition of the linear layer
  module
}

#' @export
print.tabnet_fit <- function(x, ...) {
  print(x$fit$network)
  invisible(x)
}
#' @export
print.tabnet_pretrain <- print.tabnet_fit
