add_parsnip_tabnet <- function() {

  parsnip::set_new_model("tabnet")
  parsnip::set_model_mode(model = "tabnet", mode = "classification")
  parsnip::set_model_mode(model = "tabnet", mode = "regression")

  parsnip::set_model_engine(
    "tabnet",
    mode = "classification",
    eng = "torch"
  )

  parsnip::set_model_engine(
    "tabnet",
    mode = "regression",
    eng = "torch"
  )

  parsnip::set_dependency("tabnet", eng = "torch", pkg = "tabnet")

  parsnip::set_fit(
    model = "tabnet",
    eng = "torch",
    mode = "classification",
    value = list(
      interface = "formula",
      protect = c("formula", "data", "weights"),
      func = c(pkg = "tabnet", fun = "tabnet_fit"),
      defaults = list()
    )
  )

  parsnip::set_fit(
    model = "tabnet",
    eng = "torch",
    mode = "regression",
    value = list(
      interface = "formula",
      protect = c("formula", "data", "weights"),
      func = c(pkg = "tabnet", fun = "tabnet_fit"),
      defaults = list()
    )
  )

  parsnip::set_encoding(
    model = "tabnet",
    eng = "torch",
    mode = "regression",
    options = list(
      predictor_indicators = "none",
      compute_intercept = FALSE,
      remove_intercept = FALSE,
      allow_sparse_x = FALSE
    )
  )

  parsnip::set_encoding(
    model = "tabnet",
    eng = "torch",
    mode = "classification",
    options = list(
      predictor_indicators = "none",
      compute_intercept = FALSE,
      remove_intercept = FALSE,
      allow_sparse_x = FALSE
    )
  )

  make_class_info <- function(type) {
    list(
      pre = NULL,
      post = NULL,
      func = c(fun = "predict"),
      args =
        list(
          object = quote(object$fit),
          new_data = quote(new_data),
          type = type
        )
    )
  }

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "cat_emb_dim",
    original = "cat_emb_dim",
    func = list(pkg = "dials", fun = "cat_emb_dim"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "decision_width",
    original = "decision_width",
    func = list(pkg = "dials", fun = "decision_width"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "attention_width",
    original = "attention_width",
    func = list(pkg = "dials", fun = "attention_width"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "num_steps",
    original = "num_steps",
    func = list(pkg = "dials", fun = "num_steps"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "mask_type",
    original = "mask_type",
    func = list(pkg = "dials", fun = "mask_type"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "mlp_hidden_multiplier",
    original = "mlp_hidden_multiplier",
    func = list(pkg = "dials", fun = "mlp_hidden_multiplier"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "mlp_activation",
    original = "mlp_activation",
    func = list(pkg = "dials", fun = "mlp_activation"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "encoder_activation",
    original = "encoder_activation",
    func = list(pkg = "dials", fun = "encoder_activation"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "num_independent",
    original = "num_independent",
    func = list(pkg = "dials", fun = "num_independent"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "num_shared",
    original = "num_shared",
    func = list(pkg = "dials", fun = "num_shared"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "num_independent_decoder",
    original = "num_independent_decoder",
    func = list(pkg = "dials", fun = "num_independent_decoder"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "num_shared_decoder",
    original = "num_shared_decoder",
    func = list(pkg = "dials", fun = "num_shared_decoder"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "penalty",
    original = "penalty",
    func = list(pkg = "dials", fun = "penalty"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "feature_reusage",
    original = "feature_reusage",
    func = list(pkg = "dials", fun = "feature_reusage"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "momentum",
    original = "momentum",
    func = list(pkg = "dials", fun = "momentum"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "epochs",
    original = "epochs",
    func = list(pkg = "dials", fun = "epochs"),
    has_submodel = TRUE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "batch_size",
    original = "batch_size",
    func = list(pkg = "dials", fun = "batch_size"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "virtual_batch_size",
    original = "virtual_batch_size",
    func = list(pkg = "dials", fun = "virtual_batch_size"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "learn_rate",
    original = "learn_rate",
    func = list(pkg = "dials", fun = "learn_rate"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "optimizer",
    original = "optimizer",
    func = list(pkg = "dials", fun = "optimizer"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "loss",
    original = "loss",
    func = list(pkg = "dials", fun = "loss"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "clip_value",
    original = "clip_value",
    func = list(pkg = "dials", fun = "clip_value"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "drop_last",
    original = "drop_last",
    func = list(pkg = "dials", fun = "drop_last"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "lr_scheduler",
    original = "lr_scheduler",
    func = list(pkg = "dials", fun = "lr_scheduler"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "lr_decay",
    original = "lr_decay",
    func = list(pkg = "dials", fun = "lr_decay"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "step_size",
    original = "step_size",
    func = list(pkg = "dials", fun = "step_size"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "checkpoint_epochs",
    original = "checkpoint_epochs",
    func = list(pkg = "dials", fun = "checkpoint_epochs"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "verbose",
    original = "verbose",
    func = list(pkg = "dials", fun = "verbose"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "importance_sample_size",
    original = "importance_sample_size",
    func = list(pkg = "dials", fun = "importance_sample_size"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "early_stopping_monitor",
    original = "early_stopping_monitor",
    func = list(pkg = "dials", fun = "early_stopping_monitor"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "early_stopping_tolerance",
    original = "early_stopping_tolerance",
    func = list(pkg = "dials", fun = "early_stopping_tolerance"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "early_stopping_patience",
    original = "early_stopping_patience",
    func = list(pkg = "dials", fun = "early_stopping_patience"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "skip_importance",
    original = "skip_importance",
    func = list(pkg = "dials", fun = "skip_importance"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "tabnet_model",
    original = "tabnet_model",
    func = list(pkg = "dials", fun = "tabnet_model"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "from_epoch",
    original = "from_epoch",
    func = list(pkg = "dials", fun = "from_epoch"),
    has_submodel = FALSE
  )

  parsnip::set_pred(
    model = "tabnet",
    eng = "torch",
    mode = "classification",
    type = "class",
    value = make_class_info("class")
  )

  parsnip::set_pred(
    model = "tabnet",
    eng = "torch",
    mode = "classification",
    type = "prob",
    value = make_class_info("prob")
  )

  parsnip::set_pred(
    model = "tabnet",
    eng = "torch",
    mode = "regression",
    type = "numeric",
    value = make_class_info("numeric")
  )

}

#' Parsnip compatible tabnet model
#'
#' @param mode A single character string for the type of model. Possible values
#'   for this model are "unknown", "regression", or "classification".
#' @inheritParams tabnet_config
#' @inheritParams tabnet_fit
#'
#' @inheritSection tabnet_fit Threading
#' @seealso tabnet_fit
#'
#' @examplesIf torch::torch_is_installed()
#' library(parsnip)
#' data("ames", package = "modeldata")
#' model <- tabnet() %>%
#'   set_mode("regression") %>%
#'   set_engine("torch")
#' model %>%
#'   fit(Sale_Price ~ ., data = ames)
#'
#' @return A TabNet `parsnip` instance. It can be used to fit tabnet models using
#' `parsnip` machinery.
#'
#' @export
tabnet <- function(mode = "unknown",  cat_emb_dim = NULL, decision_width = NULL, attention_width = NULL,
                   num_steps = NULL, mask_type = NULL, num_independent = NULL, num_shared = NULL,
                   num_independent_decoder = NULL, num_shared_decoder = NULL, penalty = NULL,
                   feature_reusage = NULL, momentum = NULL, epochs = NULL, batch_size = NULL,
                   virtual_batch_size = NULL, learn_rate = NULL, optimizer = NULL, loss = NULL,
                   clip_value = NULL, drop_last = NULL, lr_scheduler = NULL, lr_decay = NULL, step_size = NULL,
                   checkpoint_epochs = NULL, verbose = NULL, importance_sample_size = NULL,
                   early_stopping_monitor = NULL, early_stopping_tolerance = NULL,
                   early_stopping_patience = NULL, skip_importance = NULL,
                   tabnet_model = NULL, from_epoch = NULL
                   ) {

  if (!requireNamespace("parsnip", quietly = TRUE))
    stop("Package \"parsnip\" needed for this function to work. Please install it.", call. = FALSE)

  if (parsnip_is_missing_tabnet(tabnet_env)) {
    add_parsnip_tabnet()
    tabnet_env$parsnip_added <- TRUE
  }


  # Capture the arguments in quosures
  args <- list(
    cat_emb_dim = rlang::enquo(cat_emb_dim),
    decision_width = rlang::enquo(decision_width),
    attention_width = rlang::enquo(attention_width),
    num_steps = rlang::enquo(num_steps),
    mask_type = rlang::enquo(mask_type),
    num_independent = rlang::enquo(num_independent),
    num_shared = rlang::enquo(num_shared),
    num_independent_decoder = rlang::enquo(num_independent_decoder),
    num_shared_decoder = rlang::enquo(num_shared_decoder),
    penalty = rlang::enquo(penalty),
    feature_reusage = rlang::enquo(feature_reusage),
    momentum = rlang::enquo(momentum),
    epochs = rlang::enquo(epochs),
    batch_size = rlang::enquo(batch_size),
    virtual_batch_size = rlang::enquo(virtual_batch_size),
    learn_rate = rlang::enquo(learn_rate),
    optimizer = rlang::enquo(optimizer),
    loss = rlang::enquo(loss),
    clip_value = rlang::enquo(clip_value),
    drop_last = rlang::enquo(drop_last),
    lr_scheduler = rlang::enquo(lr_scheduler),
    lr_decay = rlang::enquo(lr_decay),
    step_size = rlang::enquo(step_size),
    checkpoint_epochs = rlang::enquo(checkpoint_epochs),
    verbose = rlang::enquo(verbose),
    importance_sample_size = rlang::enquo(importance_sample_size),
    early_stopping_monitor = rlang::enquo(early_stopping_monitor),
    early_stopping_tolerance = rlang::enquo(early_stopping_tolerance),
    early_stopping_patience = rlang::enquo(early_stopping_patience),
    skip_importance = rlang::enquo(skip_importance),
    tabnet_model = rlang::enquo(tabnet_model),
    from_epoch = rlang::enquo(from_epoch)
  )

  # Save some empty slots for future parts of the specification
  out <- list(args = args, eng_args = NULL,
              mode = mode, method = NULL, engine = "torch")

  # set classes in the correct order
  class(out) <- parsnip::make_classes("tabnet")
  out
}

tabnet_env <- new.env()
tabnet_env$parsnip_added <- FALSE

#' @export
#' @importFrom parsnip multi_predict
multi_predict._tabnet_fit <- function(object, new_data, type = NULL, epochs = NULL, ...) {

  if (is.null(epochs))
    epochs <- object$fit$config$epochs

  p <- lapply(epochs, function(epoch) {
    pred <- predict(object$fit, new_data, type = type, epoch = epoch)
    nms <- names(pred)
    pred[["epochs"]] <- epoch
    pred[[".row"]] <- 1:nrow(new_data)
    pred[, c(".row", "epochs", nms)]
  })

  p <- do.call(rbind, p)
  p <- p[order(p$.row, p$epochs),]
  p <- split(p[,-1], p$.row)
  names(p) <- NULL
  tibble::tibble(.pred = p)
}

#' @export
#' @importFrom stats update
update.tabnet <- function(object, parameters = NULL, epochs = NULL, penalty = NULL, batch_size = NULL,
                          learn_rate = NULL, decision_width = NULL, attention_width = NULL,
                          num_steps = NULL, feature_reusage = NULL, virtual_batch_size = NULL,
                          num_independent = NULL, num_shared = NULL, momentum = NULL, ...) {
  rlang::check_installed("parsnip")
  eng_args <- parsnip::update_engine_parameters(object$eng_args, fresh=TRUE, ...)
  args <- list(
    epochs = rlang::enquo(epochs),
    penalty = rlang::enquo(penalty),
    batch_size = rlang::enquo(batch_size),
    learn_rate = rlang::enquo(learn_rate),
    decision_width = rlang::enquo(decision_width),
    attention_width = rlang::enquo(attention_width),
    num_steps = rlang::enquo(num_steps),
    feature_reusage = rlang::enquo(feature_reusage),
    virtual_batch_size = rlang::enquo(virtual_batch_size),
    num_independent = rlang::enquo(num_independent),
    num_shared = rlang::enquo(num_shared),
    momentum = rlang::enquo(momentum)
  )
  args <- parsnip::update_main_parameters(args, parameters)
  parsnip::new_model_spec(
    "tabnet",
    args = args,
    eng_args = eng_args,
    mode = object$mode,
    method = NULL,
    engine = object$engine
  )
}

min_grid.tabnet <- function(x, grid, ...) tune::fit_max_value(x, grid, ...)

parsnip_is_missing_tabnet <- function(tabnet_env) {
  current <- parsnip::get_model_env()
  !(any(current$models == "tabnet") || tabnet_env$parsnip_added)
}
