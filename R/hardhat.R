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
#'   * A __Node__ where tree will be used as hierarchical outcome,
#'     and attributes will be used as predictors.
#'
#'  The predictor data should be standardized (e.g. centered or scaled).
#'  The model treats categorical predictors internally thus, you don't need to
#'  make any treatment.
#'  The model treats missing values internally thus, you don't need to make any
#'  treatment.
#'
#' @param y When `x` is a __data frame__ or __matrix__, `y` is the outcome
#' specified as:
#'
#'   * A __data frame__ with 1 or many numeric column (regression) or 1 or many categorical columns (classification) .
#'   * A __matrix__ with 1 column.
#'   * A __vector__, either numeric or categorical.
#'
#' @param data When a __recipe__ or __formula__ is used, `data` is specified as:
#'
#'   * A __data frame__ containing both the predictors and the outcome.
#'
#' @param formula A formula specifying the outcome terms on the left-hand side,
#'  and the predictor terms on the right-hand side.
#' @param tabnet_model A previously fitted `tabnet_model` object to continue the fitting on.
#'  if `NULL` (the default) a brand new model is initialized.
#' @param config A set of hyperparameters created using the `tabnet_config` function.
#'  If no argument is supplied, this will use the default values in [tabnet_config()].
#' @param from_epoch When a `tabnet_model` is provided, restore the network weights from a specific epoch.
#'  Default is last available checkpoint for restored model, or last epoch for in-memory model.
#' @param weights Unused. Placeholder for hardhat::importance_weight() variables.
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
#' @section Multi-outcome:
#'
#' TabNet allows multi-outcome prediction, which is usually named [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification)
#'   or multi-output regression when outcomes are numerical.
#' Multi-outcome currently expect outcomes to be either all numeric or all categorical.
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
#' @examplesIf (torch::torch_is_installed() && require("modeldata"))
#' \dontrun{
#' data("ames", package = "modeldata")
#' data("attrition", package = "modeldata")
#'
#' ## Single-outcome regression using formula specification
#' fit <- tabnet_fit(Sale_Price ~ ., data = ames, epochs = 4)
#'
#' ## Single-outcome classification using data-frame specification
#' attrition_x <- attrition[ids,-which(names(attrition) == "Attrition")]
#' fit <- tabnet_fit(attrition_x, attrition$Attrition, epochs = 4, verbose = TRUE)
#'
#' ## Multi-outcome regression on `Sale_Price` and `Pool_Area` in `ames` dataset using formula,
#' ames_fit <- tabnet_fit(Sale_Price + Pool_Area ~ ., data = ames, epochs = 4, valid_split = 0.2)
#'
#' ## Multi-label classification on `Attrition` and `JobSatisfaction` in
#' ## `attrition` dataset using recipe
#' library(recipes)
#' rec <- recipe(Attrition + JobSatisfaction ~ ., data = attrition) %>%
#'   step_normalize(all_numeric(), -all_outcomes())
#'
#' attrition_fit <- tabnet_fit(rec, data = attrition, epochs = 4, valid_split = 0.2)
#'
#' ## Hierarchical classification on  `acme`
#' data(acme, package = "data.tree")
#'
#' acme_fit <- tabnet_fit(acme, epochs = 4, verbose = TRUE)
#'
#' # Note: Model's number of epochs should be increased for publication-level results.
#' }
#' @return A TabNet model object. It can be used for serialization, predictions, or further fitting.
#'
#' @export
tabnet_fit <- function(x, ...) {
  UseMethod("tabnet_fit")
}

#' @export
#' @rdname tabnet_fit
tabnet_fit.default <- function(x, ...) {
  type_error("{.fn tabnet_fit} is not defined for a {.type {class(x)[1])}}.")
}

#' @export
#' @rdname tabnet_fit
tabnet_fit.data.frame <- function(x, y, tabnet_model = NULL, config = tabnet_config(), ...,
                                  from_epoch = NULL, weights = NULL) {
  processed <- hardhat::mold(x, y)
  check_type(processed$outcomes)

  config <- merge_config_and_dots(config, ...)
  tabnet_bridge(processed, config = config, tabnet_model, from_epoch, task = "supervised")
}

#' @export
#' @rdname tabnet_fit
tabnet_fit.formula <- function(formula, data, tabnet_model = NULL, config = tabnet_config(), ...,
                               from_epoch = NULL, weights = NULL) {
  processed <- hardhat::mold(
    formula, data,
    blueprint = hardhat::default_formula_blueprint(
      indicators = "none",
      intercept = FALSE
    )
  )
  check_type(processed$outcomes)

  config <- merge_config_and_dots(config, ...)
  tabnet_bridge(processed, config = config, tabnet_model, from_epoch, task = "supervised")
}

#' @export
#' @rdname tabnet_fit
tabnet_fit.recipe <- function(x, data, tabnet_model = NULL, config = tabnet_config(), ...,
                              from_epoch = NULL, weights = NULL) {
  processed <- hardhat::mold(x, data)
  check_type(processed$outcomes)

  config <- merge_config_and_dots(config, ...)
  tabnet_bridge(processed, config = config, tabnet_model, from_epoch, task = "supervised")
}

#' @export
#' @rdname tabnet_fit
#'
#' @importFrom dplyr filter mutate select mutate_all mutate_if
#' @importFrom tidyr replace_na
#'
tabnet_fit.Node <- function(x, tabnet_model = NULL, config = tabnet_config(), ..., from_epoch = NULL) {
  # ensure there is no level_* col in the Node object
  check_compliant_node(x)
  # get tree leaves and extract attributes into data.frames
  xy_df <- node_to_df(x)
  processed <- hardhat::mold(xy_df$x, xy_df$y)
  # Given n classes, M is an (n x n) matrix where M_ij = 1 if class i is descendant of class j
  ancestor <- data.tree::ToDataFrameNetwork(x) %>%
   mutate_if(is.character, ~.x %>% as.factor %>% as.numeric)
  # TODO check correctness
  # embed the M matrix in the config$ancestor variable
  dims <- c(max(ancestor), max(ancestor))
  ancestor_m <- Matrix::sparseMatrix(ancestor$from, ancestor$to, dims = dims, x = 1)
  check_type(processed$outcomes)

  config <- merge_config_and_dots(config, ...)
  tabnet_bridge(processed, config = config, tabnet_model, from_epoch, task = "supervised")
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
#'   * A __Node__ where tree leaves will be left out,
#'     and attributes will be used as predictors.
#'
#'  The predictor data should be standardized (e.g. centered or scaled).
#'  The model treats categorical predictors internally thus, you don't need to
#'  make any treatment.
#'  The model treats missing values internally thus, you don't need to make any
#'  treatment.
#'
#' @param y (optional) When `x` is a __data frame__ or __matrix__, `y` is the outcome
#' @param data When a __recipe__ or __formula__ is used, `data` is specified as:
#'
#'   * A __data frame__ containing both the predictors and the outcome.
#'
#' @param formula A formula specifying the outcome terms on the left-hand side,
#'  and the predictor terms on the right-hand side.
#' @param tabnet_model A pretrained `tabnet_model` object to continue the fitting on.
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
#' @examplesIf torch::torch_is_installed()
#' data("ames", package = "modeldata")
#' pretrained <- tabnet_pretrain(Sale_Price ~ ., data = ames, epochs = 1)
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
  type_error("{.fn tabnet_pretrain} is not defined for a {.type {class(x)[1])}}.")
}



#' @export
#' @rdname tabnet_pretrain
tabnet_pretrain.data.frame <- function(x, y = NULL, tabnet_model = NULL, config = tabnet_config(), ..., from_epoch = NULL) {
  processed <- hardhat::mold(x, y)

  config <- merge_config_and_dots(config, ...)
  tabnet_bridge(processed, config = config, tabnet_model, from_epoch, task = "unsupervised")
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
  config <- merge_config_and_dots(config, ...)
  tabnet_bridge(processed, config = config, tabnet_model, from_epoch, task = "unsupervised")
}

#' @export
#' @rdname tabnet_pretrain
tabnet_pretrain.recipe <- function(x, data, tabnet_model = NULL, config = tabnet_config(), ..., from_epoch = NULL) {
  processed <- hardhat::mold(x, data)

  config <- merge_config_and_dots(config, ...)
  tabnet_bridge(processed, config = config, tabnet_model, from_epoch, task = "unsupervised")
}

#' @export
#' @rdname tabnet_pretrain
tabnet_pretrain.Node <- function(x, tabnet_model = NULL, config = tabnet_config(), ..., from_epoch = NULL) {
  # ensure there is no level_* col in the Node object
  check_compliant_node(x)
  # get tree leaves and extract attributes into data.frames
  xy_df <- node_to_df(x)
  tabnet_pretrain(xy_df$x, tabnet_model = tabnet_model, config = config, ..., from_epoch = from_epoch)

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

#' S7 Task Classes for Double Dispatch: supervised learning task
#' @noRd
supervised_task <- S7::new_class("supervised_task", package = "tabnet")

#' S7 Task Classes for Double Dispatch: unsupervised learning task
#' @noRd
unsupervised_task <- S7::new_class("unsupervised_task", package = "tabnet")

#' S7 Task Classes for Double Dispatch: wrapper for NULL tabnet_model
#' @noRd
tabnet_null <- S7::new_class("tabnet_null", package = "tabnet")

#' S7 Prediction Classes for Double Dispatch: prediction type - numeric
#' @noRd
prediction_numeric <- S7::new_class("prediction_numeric", package = "tabnet")

#' S7 Prediction Classes for Double Dispatch: prediction type - probability
#' @noRd
prediction_prob <- S7::new_class("prediction_prob", package = "tabnet")

#' S7 Prediction Classes for Double Dispatch: prediction type - class
#' @noRd
prediction_class <- S7::new_class("prediction_class", package = "tabnet")

#' S7 Prediction Classes for Double Dispatch: single outcome
#' @noRd
single_outcome <- S7::new_class("single_outcome", package = "tabnet")

#' S7 Prediction Classes for Double Dispatch: multiple outcomes
#' @noRd
multi_outcome <- S7::new_class("multi_outcome", package = "tabnet")

#' S7 Task Classes for Double Dispatch: convert task string to S7 task object
#' @noRd
as_task <- function(task) {
  switch(task,
    supervised = supervised_task(),
    unsupervised = unsupervised_task(),
    runtime_error("Unknown task type: {.val {task}}")
  )
}

#' S7 Prediction Classes for Double Dispatch: convert prediction type to S7 object
#' @noRd
as_prediction_type <- function(type) {
  switch(type,
    numeric = prediction_numeric(),
    prob = prediction_prob(),
    class = prediction_class(),
    runtime_error("Unknown prediction type: {.val {type}}")
  )
}

#' S7 Prediction Classes for Double Dispatch: convert outcome cardinality to S7 object
#' @noRd
as_outcome_type <- function(is_multi) {
  if (is_multi) multi_outcome() else single_outcome()
}

#' S7 Task Classes for Double Dispatch: wrap tabnet_model for S7 dispatch
#'
#' Converts NULL to a tabnet_null S7 object for type-safe dispatch.
#' Non-NULL models are S3 objects (tabnet_fit or tabnet_pretrain from hardhat::new_model)
#' which S7 handles natively. Type validation happens in tabnet_bridge() before dispatch.
#'
#' @noRd
wrap_model <- function(tabnet_model) {
  if (is.null(tabnet_model)) {
    tabnet_null()
  } else {
    tabnet_model
  }
}

#' S7 Generic for tabnet_bridge_impl: double-dispatch on task and model type
#' @noRd
tabnet_bridge_impl <- S7::new_generic("tabnet_bridge_impl", dispatch_args = c("task", "model"))

#' S7 Methods Supervised Task: new supervised model (NULL -> tabnet_fit)
#' @noRd
S7::method(tabnet_bridge_impl, list(supervised_task, tabnet_null)) <- function(task, model, processed, config, epoch_shift) {
  predictors <- processed$predictors
  outcomes <- processed$outcomes

  if (sum(is.na(outcomes)) > 0) {
    value_error("Found missing values in the {.var {names(outcomes)}} outcome column.")
  }

  # new supervised model needs network initialization
  tabnet_model_lst <- tabnet_initialize(predictors, outcomes, config = config)
  tabnet_model <- new_tabnet_fit(tabnet_model_lst, blueprint = processed$blueprint)

  fit_lst <- tabnet_train_supervised(tabnet_model, predictors, outcomes, config = config, epoch_shift)
  new_tabnet_fit(fit_lst, blueprint = processed$blueprint)
}

#' S7 Methods Supervised Task: resume supervised training (tabnet_fit -> tabnet_fit)
#' @noRd
S7::method(tabnet_bridge_impl, list(supervised_task, S7::class_any)) <- function(task, model, processed, config, epoch_shift) {
  predictors <- processed$predictors
  outcomes <- processed$outcomes

  if (sum(is.na(outcomes)) > 0) {
    value_error("Found missing values in the {.var {names(outcomes)}} outcome column.")
  }

  # Default handler - check what type of model we have
  if (inherits(model, "tabnet_fit")) {
    # Resume training from supervised
    if (!check_net_is_empty_ptr(model) && inherits(model, "tabnet_fit")) {
      if (!identical(processed$blueprint, model$blueprint))
        runtime_error("Model dimensions don't match.")

      # model is available from model$serialized_net
      m <- reload_model(model$serialized_net)

      # this modifies 'model' in-place so subsequent predicts won't need to reload.
      model$fit$network$load_state_dict(m$state_dict())
      epoch_shift <- length(model$fit$metrics)

    } else if (length(model$fit$checkpoints)) {
      # model is loaded from the last available checkpoint
      last_checkpoint <- length(model$fit$checkpoints)

      model$fit$network <- reload_model(model$fit$checkpoints[[last_checkpoint]])
      epoch_shift <- last_checkpoint * model$fit$config$checkpoint_epoch

    } else {
      runtime_error("No model serialized weight can be found in {.var {model}}, check the model history")
    }

    fit_lst <- tabnet_train_supervised(model, predictors, outcomes, config = config, epoch_shift)
    return(new_tabnet_fit(fit_lst, blueprint = processed$blueprint))

  } else if (inherits(model, "tabnet_pretrain")) {
    # Transfer from pretrain to supervised
    tabnet_model_lst <- model_pretrain_to_fit(model, predictors, outcomes, config)
    tabnet_model <- new_tabnet_fit(tabnet_model_lst, blueprint = processed$blueprint)

    fit_lst <- tabnet_train_supervised(tabnet_model, predictors, outcomes, config = config, epoch_shift)
    return(new_tabnet_fit(fit_lst, blueprint = processed$blueprint))

  } else {
    type_error("Expected {.cls tabnet_fit} or {.cls tabnet_pretrain}, got {.cls {class(model)[1]}}")
  }
}

#' S7 Methods Unsupervised Task: new unsupervised pretraining (any -> tabnet_pretrain)
#' @noRd
S7::method(tabnet_bridge_impl, list(unsupervised_task, S7::class_any)) <- function(task, model, processed, config, epoch_shift) {
  predictors <- processed$predictors

  if (!S7::S7_inherits(model, tabnet_null)) {
    warn("Using {.fn tabnet_pretrain} from a model is not currently supported.",
         "Pretraining will start from a new network initialization")
  }

  pretrain_lst <- tabnet_train_unsupervised(predictors, config = config, epoch_shift)
  new_tabnet_pretrain(pretrain_lst, blueprint = processed$blueprint)
}

#' Main tabnet_bridge Function (Coordinator): double-dispatch bridge on task and model type
#'
#' Coordinates the training workflow by handling checkpoint restoration,
#' then dispatching to appropriate S7 method based on task type and model type.
#'
#' @param processed the hardhat preprocessed dataset
#' @param config the tabnet network config list of parameters
#' @param tabnet_model the tabnet model to resume training on (NULL, tabnet_fit, or tabnet_pretrain)
#' @param from_epoch the epoch to resume training from
#' @param task "supervised" or "unsupervised"
#'
#' @return a fitted tabnet_model/tabnet_pretrain object list
#' @noRd
tabnet_bridge <- function(processed, config = tabnet_config(), tabnet_model, from_epoch, task = "supervised") {
  epoch_shift <- 0L

  # Validate model type
  if (!(is.null(tabnet_model) || inherits(tabnet_model, "tabnet_fit") || inherits(tabnet_model, "tabnet_pretrain")))
    type_error("Expected NULL, {.cls tabnet_fit}, or {.cls tabnet_pretrain}, got {.cls {class(tabnet_model)[1]}}")

  # Handle checkpoint restoration (common to all paths)
  if (!is.null(from_epoch) && !is.null(tabnet_model)) {
    if (from_epoch > (length(tabnet_model$fit$checkpoints) * tabnet_model$fit$config$checkpoint_epoch))
      value_error("The model was trained for less than {.val {from_epoch}} epochs")

    # find closest checkpoint for that epoch
    closest_checkpoint <- from_epoch %/% tabnet_model$fit$config$checkpoint_epoch

    tabnet_model$fit$network <- reload_model(tabnet_model$fit$checkpoints[[closest_checkpoint]])
    epoch_shift <- closest_checkpoint * tabnet_model$fit$config$checkpoint_epoch
    tabnet_model$fit$metrics <- tabnet_model$fit$metrics[seq(epoch_shift)]
  }

  # Convert to S7 objects for double-dispatch
  task_obj <- as_task(task)
  model_obj <- wrap_model(tabnet_model)

  # Dispatch to appropriate method
  tabnet_bridge_impl(task_obj, model_obj, processed, config, epoch_shift)
}

#' S7 Generic for predict_impl_dispatch: double-dispatch on prediction type and outcome cardinality
#'
#' Replaces the previous string concatenation + switch pattern with type-safe S7 dispatch.
#' Dispatches on 2 dimensions:
#'   - Prediction type: numeric, prob, class
#'   - Outcome cardinality: single_outcome, multi_outcome
#' This creates 6 possible method combinations.
#'
#' @noRd
predict_impl_dispatch <- S7::new_generic("predict_impl_dispatch", dispatch_args = c("pred_type", "outcome_type"))

#' S7 Method: numeric prediction, single outcome
#' @noRd
S7::method(predict_impl_dispatch, list(prediction_numeric, single_outcome)) <-
  function(pred_type, outcome_type, object, predictors, batch_size, outcome_nlevels = NULL) {
    predict_impl_numeric(object, predictors, batch_size)
  }

#' S7 Method: numeric prediction, multiple outcomes
#' @noRd
S7::method(predict_impl_dispatch, list(prediction_numeric, multi_outcome)) <-
  function(pred_type, outcome_type, object, predictors, batch_size, outcome_nlevels = NULL) {
    predict_impl_numeric_multiple(object, predictors, batch_size)
  }

#' S7 Method: probability prediction, single outcome
#' @noRd
S7::method(predict_impl_dispatch, list(prediction_prob, single_outcome)) <-
  function(pred_type, outcome_type, object, predictors, batch_size, outcome_nlevels = NULL) {
    predict_impl_prob(object, predictors, batch_size)
  }

#' S7 Method: probability prediction, multiple outcomes
#' @noRd
S7::method(predict_impl_dispatch, list(prediction_prob, multi_outcome)) <-
  function(pred_type, outcome_type, object, predictors, batch_size, outcome_nlevels = NULL) {
    predict_impl_prob_multiple(object, predictors, batch_size, outcome_nlevels)
  }

#' S7 Method: class prediction, single outcome
#' @noRd
S7::method(predict_impl_dispatch, list(prediction_class, single_outcome)) <-
  function(pred_type, outcome_type, object, predictors, batch_size, outcome_nlevels = NULL) {
    predict_impl_class(object, predictors, batch_size)
  }

#' S7 Method: class prediction, multiple outcomes
#' @noRd
S7::method(predict_impl_dispatch, list(prediction_class, multi_outcome)) <-
  function(pred_type, outcome_type, object, predictors, batch_size, outcome_nlevels = NULL) {
    predict_impl_class_multiple(object, predictors, batch_size, outcome_nlevels)
  }

#' @importFrom stats predict
#' @export
predict.tabnet_fit <- function(object, new_data, type = NULL, ..., epoch = NULL) {
  if (inherits(new_data, "Node")) {
    new_data_df <- node_to_df(new_data)$x
  } else {
    new_data_df <- new_data
  }
  # Enforces column order, type, column names, etc
  processed <- hardhat::forge(new_data_df, object$blueprint)
  batch_size <- object$fit$config$batch_size
  out <- predict_tabnet_bridge(type, object, processed$predictors, epoch, batch_size)
  hardhat::validate_prediction_size(out, new_data_df)
  out
}

predict_tabnet_bridge <- function(type, object, predictors, epoch, batch_size) {

  type <- check_type(object$blueprint$ptypes$outcomes, type)
  is_multi_outcome <- ncol(object$blueprint$ptypes$outcomes) > 1
  outcome_nlevels <- NULL
  if (is_multi_outcome & type != "numeric") {
    outcome_nlevels <- purrr::map_dbl(object$blueprint$ptypes$outcomes, ~length(levels(.x)))
  }

  if (!is.null(epoch)) {

    if (epoch > (length(object$fit$checkpoints) * object$fit$config$checkpoint_epoch))
      value_error("The model was trained for less than {.val {epoch}} epochs")

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

  # Convert to S7 objects for double-dispatch
  pred_type <- as_prediction_type(type)
  outcome_type <- as_outcome_type(is_multi_outcome)

  # Dispatch to appropriate S7 method
  predict_impl_dispatch(pred_type, outcome_type, object, predictors, batch_size, outcome_nlevels)
}


model_pretrain_to_fit <- function(obj, x, y, config = tabnet_config()) {

  tabnet_model_lst <- tabnet_initialize(x, y, config)


  # do not restore previous metrics as loss function return non comparable
  # values, nor checkpoints
  m <- reload_model(obj$serialized_net)

  if (m$input_dim != tabnet_model_lst$network$input_dim)
    runtime_error("Model dimensions don't match.")

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

#' Check consistency between modeling-task type and class of outcomes vars.
#'
#' infer default modeling-task type from the outcome vars class if needed.
#'
#' @param outcome_ptype shall be `model$blueprint$ptypes$outcomes` when called from
#'  a model object, or `processed$outcomes` from the result of a `mold()`
#' @param type expected type within  `c("numeric", "prob", "class")`
#'
#' @return valid type within `c("numeric", "prob", "class")` for respectively regression,
#' class probabilities, or classification
#' @noRd
check_type <- function(outcome_ptype, type = NULL) {

  # outcome_ptype <- model$blueprint$ptypes$outcomes when called from model
  outcome_all_factor <- all(purrr::map_lgl(outcome_ptype, is.factor))
  outcome_all_numeric <- all(purrr::map_lgl(outcome_ptype, is.numeric))

  if (!outcome_all_numeric && !outcome_all_factor)
    not_implemented_error("Mixed multi-outcome type {.type {unique(purrr::map_chr(outcome_ptype, ~class(.x)[[1]]))}} is not supported")

  if (is.null(type)) {
    if (outcome_all_factor)
      type <- "class"
    else if (outcome_all_numeric)
      type <- "numeric"
    else if (ncol(outcome_ptype) == 1)
      type_error("Unknown outcome type {.type {class(outcome_ptype)}}")
  }

  type <- rlang::arg_match(type, c("numeric", "prob", "class"))

  if (outcome_all_factor) {
    if (!type %in% c("prob", "class"))
      type_error("Outcome is factor and the prediction type is {.type {type}}.")
  } else if (outcome_all_numeric) {
    if (type != "numeric")
      type_error("Outcome is numeric and the prediction type is {.type {type}}.")
  }

  invisible(type)
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

#' Prune top layer(s) of a tabnet network
#'
#' Prune `head_size` last layers of a tabnet network in order to
#'  use the pruned module as a sequential embedding module.
#' @param x nn_network to prune
#' @param head_size number of nn_layers to prune, should be less than 2
#'
#' @return a tabnet network with the top nn_layer removed
#' @rdname nn_prune_head
#' @examplesIf (torch::torch_is_installed())
#' data("ames", package = "modeldata")
#' x <- ames[,-which(names(ames) == "Sale_Price")]
#' y <- ames$Sale_Price
#' # pretrain a tabnet model on ames dataset
#' ames_pretrain <- tabnet_pretrain(x, y, epoch = 2, checkpoint_epochs = 1)
#' # prune classification head to get an embedding model
#' pruned_pretrain <- torch::nn_prune_head(ames_pretrain, 1)
#
#' @importFrom torch nn_prune_head
#' @export
nn_prune_head.tabnet_fit <- function(x, head_size) {
  if (check_net_is_empty_ptr(x)) {
    net <- reload_model(x$serialized_net)
  } else {
    net <- x$fit$network
  }
  # here we assemble nn_prune_head(x, 1) with nn_prune_head(x$tabnet, 1)
  x <- nn_prune_head(net, 1)
  x$add_module(name= "tabnet", module=nn_prune_head(net$tabnet,head_size=head_size))

}
#' @importFrom torch nn_prune_head
#' @rdname nn_prune_head
#' @export
nn_prune_head.tabnet_pretrain <- function(x, head_size) {
  if (check_net_is_empty_ptr(x)) {
    nn_prune_head(reload_model(x$serialized_net), head_size=head_size)
  } else {
    nn_prune_head(x$fit$network, head_size=head_size)
  }

}