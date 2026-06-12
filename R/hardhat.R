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
  check_type(processed$outcomes)
  
  config <- merge_config_and_dots(config, ...)
  # add ancestor boolean sparse matrix to config
  # check_dag_compliance(xy_df$y)
  config$ancestor <- build_ancestor_matrix_from_outcomes(x, processed$outcomes)
  # make outcomes levels available so that batched y could be one-hot encoded.
  config$outcomes <- processed$outcomes
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

#' Triple dispatch on task, resume training and resume epoch
#'
#' Perform the triple dispatch and initialize the model (if needed) or
#'  resume the model network weight to the right epoch
#'
#' @param processed the hardhat prerocessed dataset
#' @param config the tabnet network config list of parameters
#' @param tabnet_model the tabnet model to resume training on
#' @param from_epoch the epoch to resume training from
#' @param task "supervised" or "unsupervised"
#'
#' @return a fitted tabnet_model/tabnet_pretrain object list
#' @noRd
tabnet_bridge <- function(processed, config = tabnet_config(), tabnet_model, from_epoch, task="supervised") {
  predictors <- processed$predictors
  outcomes <- processed$outcomes
  epoch_shift <- 0L

  if (!(is.null(tabnet_model) || inherits(tabnet_model, "tabnet_fit") || inherits(tabnet_model, "tabnet_pretrain")))
    type_error("{.var {tabnet_model}} is not recognised as a proper TabNet model")

  if (!is.null(from_epoch) && !is.null(tabnet_model)) {
    # model must be loaded from checkpoint

    if (from_epoch > (length(tabnet_model$fit$checkpoints) * tabnet_model$fit$config$checkpoint_epoch))
      value_error("The model was trained for less than {.val {from_epoch}} epochs")

    # find closest checkpoint for that epoch
    closest_checkpoint <- from_epoch %/% tabnet_model$fit$config$checkpoint_epoch

    tabnet_model$fit$network <- reload_model(tabnet_model$fit$checkpoints[[closest_checkpoint]])
    epoch_shift <- closest_checkpoint * tabnet_model$fit$config$checkpoint_epoch
    tabnet_model$fit$metrics <- tabnet_model$fit$metrics[seq(epoch_shift)]

  }
  if (task == "supervised") {
    if (sum(is.na(outcomes)) > 0) {
      value_error("Found missing values in the {.var {names(outcomes)}} outcome column.")
    }
    if (is.null(tabnet_model)) {
      # new supervised model needs network initialization
      tabnet_model_lst <- tabnet_initialize(predictors, outcomes, config = config)
      tabnet_model <-  new_tabnet_fit(tabnet_model_lst, blueprint = processed$blueprint)

    } else if (!check_net_is_empty_ptr(tabnet_model) && inherits(tabnet_model, "tabnet_fit")) {
      # resume training from supervised
      if (!identical(processed$blueprint, tabnet_model$blueprint))
        runtime_error("Model dimensions don't match.")

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

    } else runtime_error("No model serialized weight can be found in {.var {tabnet_model}}, check the model history")

    fit_lst <- tabnet_train_supervised(tabnet_model, predictors, outcomes, config = config, epoch_shift)
    return(new_tabnet_fit(fit_lst, blueprint = processed$blueprint))

  } else if (task == "unsupervised") {

    if (!is.null(tabnet_model)) {
      warn("Using {.fn tabnet_pretrain} from a model is not currently supported.",
           "Pretraining will start from a new network initialization")
    }
    pretrain_lst <- tabnet_train_unsupervised( predictors, config = config, epoch_shift)
    return(new_tabnet_pretrain(pretrain_lst, blueprint = processed$blueprint))

  }
}


#' Predict using `tabnet`
#'
#' @param object,x A `tabnet_fit` object.
#'
#' @param new_data A data frame or matrix of new predictors.
#' @param type expected outcome type within  `c("numeric", "prob", "class")`.
#' @param epoch the epoch of an existing checkpoint to infer from.
#' 
#' @param ... Not used, but required for extensibility.
#'
#' @return
#'
#' [predict()] returns a tibble of predictions and [augment()] appends the
#' columns in `new_data`. In either case, the number of rows in the tibble is
#' guaranteed to be the same as the number of rows in `new_data`.
#'
#' For regression data, the prediction is in the column `.pred`. For
#' classification, the class predictions are in `.pred_class` and the
#' probability estimates are in columns with the pattern `.pred_{level}` where
#' `level` is the levels of the outcome factor vector.
#'
#' @examples
#' # Minimal example for quick execution
#' car_split <- rsample::initial_split(mtcars[ 1:6,   ])
#'
#' \dontrun{
#' # Fit
#' if (torch_is_installed() & interactive()) {
#'  mod <- tabnet_fit(mpg ~ cyl + log(drat), training(car_split))
#'
#'  # Predict
#'  predict(mod, testing(car_split))
#'  augment(mod, testing(car_split))
#' }
#' }
#'
#' @importFrom stats predict
#' @export
predict.tabnet_fit <- function(object, new_data, type = NULL, ..., epoch = NULL) {
  if (inherits(new_data, "Node") && !is.null(object$fit$config$ancestor)) {
    new_data_df <- node_to_df(new_data)$x
    # Enforces column order, type, column names, etc
    processed <- hardhat::forge(new_data_df, object$blueprint)
    
  } else {
    new_data_df <- new_data
    processed <- hardhat::forge(new_data, object$blueprint)
  }
  batch_size <- object$fit$config$batch_size
  out <- predict_tabnet_bridge(type, object, processed$predictors, epoch, batch_size)
  hardhat::validate_prediction_size(out, new_data_df)
  out
}

#' @export
#' @inheritParams predict.tabnet_fit
#' @rdname predict.tabnet_fit
augment.tabnet_fit <- function(x, new_data, ...) {
  res <- predict(x, new_data, ...)
  if (inherits(new_data, "Node") && !is.null(x$fit$config$ancestor)) {
    new_data_df <- node_to_df(new_data)
    # Enforces column order, type, outcomes column names, etc
    forged_truth <- hardhat::forge(cbind(new_data_df$x, new_data_df$y), x$blueprint, outcomes = TRUE)$outcomes
  } else {
    # mold XY blueprint
    # When mold() was called with a vector y, hardhat uses ".outcome" as the outcome column
    # name. forge() with outcomes = TRUE then requires new_data to contain ".outcome", which
    # won't be the case when the user passes a regular data frame.
    if (inherits(x$blueprint, "xy_blueprint") && ncol(x$blueprint$ptypes$outcomes) == 1) {
      outcome_name_col <- which(!names(new_data) %in% names(x$blueprint$ptypes$predictors))
      names(new_data)[outcome_name_col] <- ".outcome"
    } 
    forged_truth <- hardhat::forge(new_data, blueprint = x$blueprint, outcomes = TRUE)$outcomes
  }
  dplyr::bind_cols(res, forged_truth)
}


predict_tabnet_bridge <- function(type, object, predictors, epoch, batch_size) {

  type <- check_type(object$blueprint$ptypes$outcomes, type)
  is_multi_outcome <- ncol(object$blueprint$ptypes$outcomes) > 1
  outcome_nlevels <- NULL
  if (is_multi_outcome && type != "numeric") {
    outcome_nlevels <- purrr::map_dbl(object$blueprint$ptypes$outcomes, ~nlevels(.x))
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

  
  type_multioutcome <- paste0(type, "_", is_multi_outcome)
  switch(
    type_multioutcome,
    numeric_FALSE = predict_impl_numeric(object, predictors, batch_size),
    numeric_TRUE  = predict_impl_numeric_multiple(object, predictors, batch_size),
    prob_FALSE    = predict_impl_prob(object, predictors, batch_size),
    prob_TRUE     = predict_impl_prob_multiple(object, predictors, batch_size, outcome_nlevels),
    class_FALSE   = predict_impl_class(object, predictors, batch_size),
    class_TRUE    = predict_impl_class_multiple(object, predictors, batch_size, outcome_nlevels)
  )
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
#' @param type expected outcome type within  `c("numeric", "prob", "class")`
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


#' Build ancestor matrix aligned with observed outcome classes
#'
#' Extracts class names from the outcome tibble (factor levels) and builds
#' the ancestor matrix only for classes that actually appear in the data.
#'
#' @param x A `data.tree::Node` object.
#' @param outcomes A tibble with factor columns (one per hierarchy level),
#'   as returned by `hardhat::mold()$outcomes`.
#' @param device Torch device ("cpu" or "cuda").
#' @return A `torch_tensor` of shape `(1, n_classes, n_classes)`.
#' @export
build_ancestor_matrix_from_outcomes <- function(x, outcomes, device = "cpu") {
  # 1. Extract all class names from factor levels (preserving order)
  #    outcomes is a tibble with one factor column per hierarchy level
  level_cols <- names(outcomes)
  all_class_names <- unlist(lapply(outcomes, levels), use.names = FALSE)
  n_classes <- length(all_class_names)
  
  if (n_classes == 0L) {
    runtime_error("No factor levels found in outcomes : {str(outcomes)}")
  }
  
  # 2. Build a lookup: class_name -> data.tree Node
  all_nodes <- data.tree::Traverse(x, traversal = "pre-order")
  all_nodes <- unname(all_nodes)
  level_lengths <- lengths(lapply(outcomes, levels))
  lvl_vector <- rep(seq_along(level_cols) + 1L, level_lengths)
  
  # 3. Resolve each class name to its Node
  class_nodes <- lapply(seq_along(all_class_names), function(k) {
    nm <- all_class_names[k]
    lvl <- lvl_vector[k]
    
    candidates <- Filter(function(n) n$level == lvl && n$name == nm, all_nodes)
    if (length(candidates) == 0) {
      runtime_error("Factor level {.var {nm}} not found at tree level {lvl} (outcomes column {.var {level_cols[lvl - 1L]}})")
    }
    candidates[[1]]
  })
  
  # 4. Create 1-based index mapping
  class_map <- setNames(seq_len(n_classes), all_class_names)
  
  # 5. Collect (descendant, ancestor) pairs by climbing up
  row_list <- vector("list", n_classes)
  col_list <- vector("list", n_classes)
  
  for (i in seq_len(n_classes)) {
    current <- class_nodes[[i]]
    anc_indices <- integer()
    
    repeat {
      idx <- class_map[current$name]
      if (!is.null(idx)) {
        anc_indices <- c(anc_indices, idx)
      }
      if (current$isRoot || is.null(current$parent)) break
      current <- current$parent
    }
    
    row_list[[i]] <- rep(i, length(anc_indices))
    col_list[[i]] <- anc_indices
  }
  
  # 6. Fill matrix
  R <- matrix(0L, nrow = n_classes, ncol = n_classes)
  rows <- unlist(row_list, use.names = FALSE)
  cols <- unlist(col_list, use.names = FALSE)
  if (length(rows) > 0) R[cbind(rows, cols)] <- 1L
  
  # 7. Convert to torch
  R_torch <- torch::torch_tensor(R, device = device)
  R_torch$unsqueeze(1)
}

