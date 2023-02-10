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
      protect = c("formula", "data"),
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
      protect = c("formula", "data"),
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
    parsnip = "penalty",
    original = "penalty",
    func = list(pkg = "dials", fun = "penalty"),
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
    parsnip = "decision_width",
    original = "decision_width",
    func = list(pkg = "tabnet", fun = "decision_width"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "attention_width",
    original = "attention_width",
    func = list(pkg = "tabnet", fun = "attention_width"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "num_steps",
    original = "num_steps",
    func = list(pkg = "tabnet", fun = "num_steps"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "feature_reusage",
    original = "feature_reusage",
    func = list(pkg = "tabnet", fun = "feature_reusage"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "virtual_batch_size",
    original = "virtual_batch_size",
    func = list(pkg = "dials", fun = "batch_size"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "num_independent",
    original = "num_independent",
    func = list(pkg = "tabnet", fun = "num_independent"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "num_shared",
    original = "num_shared",
    func = list(pkg = "tabnet", fun = "num_shared"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "tabnet",
    eng = "torch",
    parsnip = "momentum",
    original = "momentum",
    func = list(pkg = "tabnet", fun = "momentum"),
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
#'
#' @inheritSection tabnet_fit Threading
#' @seealso tabnet_fit
#'
#' @examples
#' if (torch::torch_is_installed()) {
#' library(parsnip)
#' data("ames", package = "modeldata")
#' model <- tabnet() %>%
#'   set_mode("regression") %>%
#'   set_engine("torch")
#' model %>%
#'   fit(Sale_Price ~ ., data = ames)
#' }
#'
#' @return A TabNet `parsnip` instance. It can be used to fit tabnet models using
#' `parsnip` machinery.
#'
#' @export
tabnet <- function(mode = "unknown", epochs = NULL, penalty = NULL, batch_size = NULL,
                   learn_rate = NULL, decision_width = NULL, attention_width = NULL,
                   num_steps = NULL, feature_reusage = NULL, virtual_batch_size = NULL,
                   num_independent = NULL, num_shared = NULL, momentum = NULL) {

  if (!requireNamespace("parsnip", quietly = TRUE))
    rlang::abort("Package \"parsnip\" needed for this function to work. Please install it.")

  if (!tabnet_env$parsnip_added) {
    add_parsnip_tabnet()
    tabnet_env$parsnip_added <- TRUE
  }

  # Capture the arguments in quosures
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

  # Save some empty slots for future parts of the specification
  out <- list(args = args, eng_args = NULL,
              mode = mode, method = NULL, engine = "torch")

  # set classes in the correct order
  class(out) <- parsnip::make_classes("tabnet")
  out
}

tabnet_env <- new.env()
tabnet_env$parsnip_added <- FALSE


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
