merge_config_and_dots <- function(config, ...) {
  default_config <- tabnet_config()
  new_config <- do.call(tabnet_config, list(...))
  # TODO currently we cannot not compare two nn_optimizer nor nn_loss values 
  new_config <- new_config[
    mapply(
      function(x, y) ifelse(is_null_or_optim_generator_or_loss(x), 
                            !is_null_or_optim_generator_or_loss(y), # TRUE 
                            ifelse(is_optim_generator_or_loss(y), TRUE, x != y)), # FALSE 
      default_config,
      new_config)
  ]
  merged_config <- utils::modifyList(config, as.list(new_config))
  merged_config$optimizer <- resolve_optimizer(merged_config$optimizer)
  merged_config
}
# 
# is_different_param <- function(x, y) {
#   if (rlang::inherits_any(x, c("nn_loss", "nn_optim_generatorclass"))) {
#     
#   }
# }

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

#' Check that Node object names are compliant
#'
#' @param node the Node object, or a dataframe ready to be parsed by `data.tree::as.Node()`
#'
#' @return node if it is compliant, else an Error with the column names to fix
#' @export
#'
#' @examplesIf (require("data.tree") || require("dplyr"))
#' library(dplyr)
#' library(data.tree)
#' data(starwars)
#' starwars_tree <- starwars %>%
#'   mutate(pathString = paste("tree", species, homeworld, `name`, sep = "/"))
#'
#' # pre as.Node() check
#' try(check_compliant_node(starwars_tree))
#'
#' # post as.Node() check
#' check_compliant_node(as.Node(starwars_tree))
#'
check_compliant_node <- function(node) {
  #  prevent reserved data.tree Node colnames and the level_1 ... level_n names used for coercion
  if (inherits(node, "Node")) {
    # Node has already lost its reserved colnames
    reserved_names <- paste0("level_", c(1:node$height))
    actual_names <- node$attributesAll
  } else if (inherits(node, "data.frame") && "pathString" %in% colnames(node)) {
    node_height <- max(stringr::str_count(node$pathString, "/"))
    reserved_names <- c(paste0("level_", c(1:node_height)), data.tree::NODE_RESERVED_NAMES_CONST)
    actual_names <- colnames(node)[!colnames(node) %in% "pathString"]
  } else {
    type_error("The provided hierarchical object is not recognized with a valid format that can be checked")
  }
  
  if (any(actual_names %in% reserved_names)) {
    value_error("The attributes or colnames in the provided hierarchical object use the following reserved names:
                {.vars {actual_names[actual_names %in% reserved_names]}}. 
                Please change those names as they will lead to unexpected tabnet behavior.")
  }
  
  invisible(node)
}

#' Turn a Node object into predictor and outcome.
#'
#' @param x Node object
#' @param drop_last_level TRUE unused
#'
#' @return a named list of x and y, being respectively the predictor data-frame and the outcomes data-frame,
#'   as expected inputs for `hardhat::mold()` function.
#' @export
#'
#' @examplesIf (require("data.tree") || require("dplyr"))
#' library(dplyr)
#' library(data.tree)
#' data(starwars)
#' starwars_tree <- starwars %>%
#'   mutate(pathString = paste("tree", species, homeworld, `name`, sep = "/")) %>%
#'   as.Node()
#' node_to_df(starwars_tree)$x %>% head()
#' node_to_df(starwars_tree)$y %>% head()
#' @importFrom dplyr last_col mutate mutate_if select starts_with where
node_to_df <- function(x, drop_last_level = TRUE) {
  # TODO get rid of all those import through base R equivalent
  xy_df <- data.tree::ToDataFrameTypeCol(x, x$attributesAll)
  x_df <- xy_df %>%
    select(-starts_with("level_")) %>%
    mutate_if(is.character, as.factor)
  y_df <- xy_df %>%
    select(starts_with("level_")) %>%
    # drop first (and all zero-variance) column
    select(where(~ nlevels(as.factor(.x)) > 1 )) %>%
    # TODO take the drop_last_level param into account
    # drop last level column
    select(-last_col()) %>%
    # TODO impute "NA" with parent through coalesce() via an option
    mutate_if(is.character, as.factor)
  return(list(x = x_df, y = y_df))
}


model_to_raw <- function(model) {
  con <- rawConnection(raw(), open = "wr")
  torch::torch_save(model, con)
  on.exit({close(con)}, add = TRUE)
  r <- rawConnectionValue(con)
  r
}

# generalize torch to_device to nested list of tensors
to_device <- function(x, device) {
  lapply(x, function(x) {
    if (inherits(x, "torch_tensor")) {
      x$to(device=device)
    } else if (is.list(x)) {
      lapply(x, to_device)
    } else {
      x
    }
  })
}

# `optim_ignite_*` requires a minimum torch version
torch_has_optim_ignite <- function() {
  utils::compareVersion(as.character(utils::packageVersion("torch")), "0.14.0") >= 0
}

# turn "adam" or a torch_optim_* generator into a proper torch_optim_ generator
resolve_optimizer <- function(optimizer) {
  if (is_optim_generator(optimizer)) {
    torch_optimizer <- optimizer
  } else if (is.character(optimizer)) {
    if (optimizer == "adam" && torch_has_optim_ignite()) {
      torch_optimizer <- torch::optim_ignite_adam
    } else if (optimizer == "adam") {
      torch_optimizer <- torch::optim_adam
    } else {
      value_error("Currently only {.val adam} is supported as character for {.var optimizer}.")
    }
  } else {
    value_error("Currently only {.val adam} is supported as character for {.var optimizer}.")
  }
  torch_optimizer
  
}


is_optim_generator <- function(x) {
  inherits(x, "torch_optimizer_generator")
}

is_loss_generator <- function(x) {
  rlang::inherits_all(x, c("nn_loss", "nn_module_generator"))
}

is_null_or_optim_generator_or_loss <- function(x) {
  is.null(x) || is_optim_generator(x) || inherits(x, "nn_loss")
}

is_optim_generator_or_loss <- function(x) {
  is_optim_generator(x) || inherits(x, "nn_loss")
}


value_error <- function(..., env = rlang::caller_env()) {
  cli::cli_abort(gettext(..., domain = "R-tabnet")[[1]], .envir = env)
}

type_error <- function(..., env = rlang::caller_env()) {
  cli::cli_abort(gettext(..., domain = "R-tabnet")[[1]], .envir = env)
}

runtime_error <- function(..., env = rlang::caller_env()) {
  cli::cli_abort(gettext(..., domain = "R-tabnet")[[1]], .envir = env)
}

not_implemented_error <- function(..., env = rlang::caller_env()) {
  cli::cli_abort(gettext(..., domain = "R-tabnet")[[1]], .envir = env)
}

warn <- function(..., env = rlang::caller_env()) {
  cli::cli_warn(gettext(..., domain = "R-tabnet")[[1]], .envir = env)
}
