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
#' @param config A set of hyperparameters created using the `tabnet_config` function.
#'  If no argument is supplied, this will use the default values in [tabnet_config()].
#' @param from_epoch When a `tabnet_model` is provided, restore the network weights from a specific epoch.
#'  Default is last available checkpoint for restored model, or last epoch for in-memory model.
#' @param ... Model hyperparameters.
#' Any hyperparameters set here will update those set by the config argument.
#' See [tabnet_config()] for a list of all possible hyperparameters.
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
tabnet_fit.data.frame <- function(x, y, tabnet_model = NULL, config = tabnet_config(), ..., from_epoch = NULL) {
  processed <- hardhat::mold(x, y)

  default_config <- tabnet_config()
  new_config <- do.call(tabnet_config, list(...))
  new_config <- new_config[
    mapply(
    function(x, y) ifelse(is.null(x), !is.null(y), x != y),
    default_config,
    new_config)
    ]
  config <- utils::modifyList(config, as.list(new_config))

  tabnet_bridge(processed, config = config, tabnet_model, from_epoch, task="supervised")
}

#' @export
#' @rdname tabnet_fit
tabnet_fit.formula <- function(formula, data, tabnet_model = NULL, config = tabnet_config(), ..., from_epoch = NULL) {
  processed <- hardhat::mold(
    formula, data,
    blueprint = hardhat::default_formula_blueprint(
      indicators = "none",
      intercept = FALSE
    )
  )

  default_config <- tabnet_config()
  new_config <- do.call(tabnet_config, list(...))
  new_config <- new_config[
    mapply(
      function(x, y) ifelse(is.null(x), !is.null(y), x != y),
      default_config,
      new_config)
  ]
  config <- utils::modifyList(config, as.list(new_config))

  tabnet_bridge(processed, config = config, tabnet_model, from_epoch, task="supervised")
}

#' @export
#' @rdname tabnet_fit
tabnet_fit.recipe <- function(x, data, tabnet_model = NULL, config = tabnet_config(), ..., from_epoch = NULL) {
  processed <- hardhat::mold(x, data)

  default_config <- tabnet_config()
  new_config <- do.call(tabnet_config, list(...))
  new_config <- new_config[
    mapply(
      function(x, y) ifelse(is.null(x), !is.null(y), x != y),
      default_config,
      new_config)
  ]
  config <- utils::modifyList(config, as.list(new_config))

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
#' @param config A set of hyperparameters created using the `tabnet_config` function.
#'  If no argument is supplied, this will use the default values in [tabnet_config()].
#' @param from_epoch When a `tabnet_model` is provided, restore the network weights from a specific epoch.
#'  Default is last available checkpoint for restored model, or last epoch for in-memory model.
#' @param ... Model hyperparameters.
#' Any hyperparameters set here will update those set by the config argument.
#' See [tabnet_config()] for a list of all possible hyperparameters.
#'
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
tabnet_pretrain.data.frame <- function(x, y, tabnet_model = NULL, config = tabnet_config(), ..., from_epoch = NULL) {
  processed <- hardhat::mold(x, y)

  default_config <- tabnet_config()
  new_config <- do.call(tabnet_config, list(...))
  new_config <- new_config[
    mapply(
      function(x, y) ifelse(is.null(x), !is.null(y), x != y),
      default_config,
      new_config)
  ]
  config <- utils::modifyList(config, as.list(new_config))

  tabnet_bridge(processed, config = config, tabnet_model, from_epoch, task="unsupervised")
}

#' @export
#' @rdname tabnet_pretrain
tabnet_pretrain.formula <- function(formula, data, tabnet_model = NULL, config = tabnet_config(), ..., from_epoch = NULL) {
  processed <- hardhat::mold(
    formula, data,
    blueprint = hardhat::default_formula_blueprint(
      indicators = "none",
      intercept = FALSE
    )
  )

  default_config <- tabnet_config()
  new_config <- do.call(tabnet_config, list(...))
  new_config <- new_config[
    mapply(
      function(x, y) ifelse(is.null(x), !is.null(y), x != y),
      default_config,
      new_config)
  ]
  config <- utils::modifyList(config, as.list(new_config))

  tabnet_bridge(processed, config = config, tabnet_model, from_epoch, task="unsupervised")
}

#' @export
#' @rdname tabnet_pretrain
tabnet_pretrain.recipe <- function(x, data, tabnet_model = NULL, config = tabnet_config(), ..., from_epoch = NULL) {
  processed <- hardhat::mold(x, data)

  default_config <- tabnet_config()
  new_config <- do.call(tabnet_config, list(...))
  new_config <- new_config[
    mapply(
      function(x, y) ifelse(is.null(x), !is.null(y), x != y),
      default_config,
      new_config)
  ]
  config <- utils::modifyList(config, as.list(new_config))

  tabnet_bridge(processed, config = config, tabnet_model, from_epoch, task="unsupervised")
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
  epoch_shift <- 0L
  if (!(is.null(tabnet_model) || inherits(tabnet_model, "tabnet_fit") || inherits(tabnet_model, "tabnet_pretrain")))
    rlang::abort(paste0(tabnet_model," is not recognised as a proper TabNet model"))

  if (!is.null(from_epoch) && !is.null(tabnet_model)) {
    # model must be loaded from checkpoint

    if (from_epoch > (length(tabnet_model$fit$checkpoints) * tabnet_model$fit$config$checkpoint_epoch))
      rlang::abort(paste0("The model was trained for less than ", from_epoch, " epochs"))

    # find closest checkpoint for that epoch
    closest_checkpoint <- from_epoch %/% tabnet_model$fit$config$checkpoint_epoch

    tabnet_model$fit$network <- reload_model(tabnet_model$fit$checkpoints[[closest_checkpoint]])
    epoch_shift <- closest_checkpoint * tabnet_model$fit$config$checkpoint_epoch
    tabnet_model$fit$metrics <- tabnet_model$fit$metrics[seq(epoch_shift)]

  }
  if (task == "supervised") {
    if (is.null(tabnet_model)) {
      # new supervised model needs network initialization
      tabnet_model_lst <- tabnet_initialize(predictors, outcomes, config = config)
      tabnet_model <-  new_tabnet_fit(tabnet_model_lst, blueprint = processed$blueprint)

    } else if (!check_net_is_empty_ptr(tabnet_model) && inherits(tabnet_model, "tabnet_fit")) {
      # resume training from supervised
      if (!identical(processed$blueprint, tabnet_model$blueprint))
        rlang::abort("Model dimensions don't match.")

      # model is available from tabnet_model$serialized_net
      m <- reload_model(tabnet_model$serialized_net)

      # this modifies 'tabnet_model' in-place so subsequent predicts won't
      # need to reload.
      tabnet_model$fit$network$load_state_dict(m$state_dict())
      epoch_shift <- length(tabnet_model$fit$metrics)


    } else if (inherits(tabnet_model, "tabnet_pretrain")) {
      # resume training from unsupervised

      tabnet_model_lst <- model_pretrain_to_fit(tabnet_model, predictors, outcomes, config)
      tabnet_model <-  new_tabnet_fit(tabnet_model_lst, blueprint = processed$blueprint)

    }  else if (length(tabnet_model$fit$checkpoints)) {
      # model is loaded from the last available checkpoint

      last_checkpoint <- length(tabnet_model$fit$checkpoints)

      tabnet_model$fit$network <- reload_model(tabnet_model$fit$checkpoints[[last_checkpoint]])
      epoch_shift <- last_checkpoint * tabnet_model$fit$config$checkpoint_epoch

    } else rlang::abort(paste0("No model serialized weight can be found in ", tabnet_model, ", check the model history"))

    fit_lst <- tabnet_train_supervised(tabnet_model, predictors, outcomes, config = config, epoch_shift)
    return(new_tabnet_fit(fit_lst, blueprint = processed$blueprint))

  } else if (task == "unsupervised") {

    pretrain_lst <- tabnet_train_unsupervised( predictors, config = config, epoch_shift)
    return(new_tabnet_pretrain(pretrain_lst, blueprint = processed$blueprint))

  }
}


#' @importFrom stats predict
#' @export
predict.tabnet_fit <- function(object, new_data, type = NULL, ..., epoch = NULL) {
  # Enforces column order, type, column names, etc
  processed <- hardhat::forge(new_data, object$blueprint)
  out <- predict_tabnet_bridge(type, object, processed$predictors, epoch, batch_size)
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



predict_tabnet_bridge <- function(type, object, predictors, epoch, batch_size) {

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
    numeric = predict_impl_numeric(object, predictors, batch_size),
    prob    = predict_impl_prob(object, predictors, batch_size),
    class   = predict_impl_class(object, predictors, batch_size)
  )
}

model_to_raw <- function(model) {
  con <- rawConnection(raw(), open = "wr")
  torch::torch_save(model, con)
  on.exit({close(con)}, add = TRUE)
  r <- rawConnectionValue(con)
  r
}

model_pretrain_to_fit <- function(obj, x, y, config = tabnet_config()) {

  tabnet_model_lst <- tabnet_initialize(x, y, config)


  # do not restore previous metrics as loss function return non comparable
  # values, nor checkpoints
  m <- reload_model(obj$serialized_net)

  if (m$input_dim != tabnet_model_lst$network$input_dim)
    rlang::abort("Model dimensions don't match.")

  # perform update of selected weights into new tabnet_model
  m_stat_dict <- m$state_dict()
  tabnet_state_dict <- tabnet_model_lst$network$state_dict()
  for (param in names(m_stat_dict)) {
    if (grepl("^encoder", param)) {
      # Convert encoder's layers name to match
      new_param <- paste0("tabnet.", param)
    } else {
      new_param <- param
    }
    if (!is.null(tabnet_state_dict[new_param])) {
      tabnet_state_dict[[new_param]] <- m_stat_dict[[param]]
    }
  }
  tabnet_model_lst$network$load_state_dict(tabnet_state_dict)
  tabnet_model_lst
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
  if (check_net_is_empty_ptr(x)) {
    print(reload_model(x$serialized_net))
  } else {
    print(x$fit$network)
  }
  invisible(x)
}
#' @export
print.tabnet_pretrain <- print.tabnet_fit
