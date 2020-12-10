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
#' @export
tabnet <- function(mode = "unknown", epochs = NULL) {

  if (!tabnet_env$parsnip_added) {
    add_parsnip_tabnet()
    tabnet_env$parsnip_added <- TRUE
  }

  # Capture the arguments in quosures
  args <- list(epochs = rlang::enquo(epochs))

  # Save some empty slots for future parts of the specification
  out <- list(args = args, eng_args = NULL,
              mode = mode, method = NULL, engine = NULL)

  # set classes in the correct order
  class(out) <- parsnip::make_classes("tabnet")
  out
}

tabnet_env <- new.env()
tabnet_env$parsnip_added <- FALSE


