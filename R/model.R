
#' Transforms input data into tensors
#'
#' @param x a data frame
#' @param y a response vector
resolve_data <- function(x, y) {

  # convert factors to integers
  x_ <- x
  for (v in seq_along(x_)) {
    if (is.factor(x_[[v]]))
      x_[[v]] <- as.numeric(x_[[v]])
  }
  x_tensor <- torch::torch_tensor(as.matrix(x_), dtype = torch::torch_float())

  if (ncol(y) == 1)
    y <- y[[1]]

  if (is.factor(y)) {
    y_tensor <- torch::torch_tensor(as.integer(y), dtype = torch::torch_int64())
  } else {
    y_tensor <- torch::torch_tensor(y, dtype = torch::torch_float())$unsqueeze(2)
  }

  cat_idx <- which(sapply(x, is.factor))

  if (is.factor(y))
    output_dim <- max(as.integer(y))
  else
    output_dim <- 1

  input_dim <- ncol(x)

  list(x = x_tensor, y = y_tensor, cat_idx = cat_idx, output_dim = output_dim,
       input_dim = input_dim)
}

#' Configuration for TabNet models
#'
#' @param batch_size (int) Number of examples per batch, large batch sizes are
#'   recommended. (default: 1024)
#' @param lambda_sparse This is the extra sparsity loss coefficient as proposed
#'   in the original paper. The bigger this coefficient is, the sparser your model
#'   will be in terms of feature selection. Depending on the difficulty of your
#'   problem, reducing this value could help.
#' @param clip_value (NULL) If a float is given this will clip the gradient at
#'   clip_value.
#' @param loss (character or function) Loss function for training (default to mse
#'   for regression and cross entropy for classification)
#' @param epochs (int) Number of training epochs.
#' @param drop_last (bool) Whether to drop last batch if not complete during
#'   training
#' @param n_d (int) Width of the decision prediction layer. Bigger values gives
#'   more capacity to the model with the risk of overfitting. Values typically
#'   range from 8 to 64.
#' @param n_a (int) Width of the attention embedding for each mask. According to
#'   the paper n_d=n_a is usually a good choice. (default=8)
#' @param n_steps (int) Number of steps in the architecture
#'   (usually between 3 and 10)
#' @param gamma (float) This is the coefficient for feature reusage in the masks.
#'   A value close to 1 will make mask selection least correlated between layers.
#'   Values range from 1.0 to 2.0.
#' @param virtual_batch_size (int) Size of the mini batches used for
#'   "Ghost Batch Normalization" (default=128)
#' @param valid_split (float) The fraction of the dataset used for validation.
#' @param verbose (bool) wether to print progress and loss values during
#'   training.
#'
#' @export
tabnet_config <- function(...) {
  default <- list(
    batch_size = 256,
    lambda_sparse = 1e-3,
    clip_value = 1,
    loss = "auto",
    epochs = 5,
    drop_last = FALSE,
    n_d = 8,
    n_a = 8,
    n_steps = 3,
    gamma = 1.3,
    virtual_batch_size = 128,
    valid_split = 0,
    verbose = FALSE
  )
  args <- list(...)
  for (arg_nm in names(args)) {

    if (is.null(default[[arg_nm]]))
      rlang::abort(paste0("Argument '", arg_nm, "' is not an hyperparameter."))

    default[[arg_nm]] <- args[[arg_nm]]
  }
  default
}

train_batch <- function(network, optimizer, batch, config) {
  # forward pass
  output <- network(batch$x)
  loss <- config$loss_fn(output[[1]], batch$y)

  # Add the overall sparsity loss
  loss <- loss - config$lambda_sparse * output[[2]]

  # step of the optimization
  optimizer$zero_grad()
  loss$backward()
  if (!is.null(config$clip_value)) {
    torch::nn_utils_clip_grad_norm_(network$parameters, config$clip_value)
  }
  optimizer$step()

  list(
    loss = loss$item()
  )
}

valid_batch <- function(network, batch, config) {
  # forward pass
  output <- network(batch$x)
  loss <- config$loss_fn(output[[1]], batch$y)

  # Add the overall sparsity loss
  loss <- loss - config$lambda_sparse * output[[2]]

  list(
    loss = loss$item()
  )
}

transpose_metrics <- function(metrics) {
  nms <- names(metrics[1])
  out <- vector(mode = "list", length = length(nms))
  for (nm in nms) {
    out[[nm]] <- vector("numeric", length = length(metrics))
  }

  for (i in seq_along(metrics)) {
    for (nm in nms) {
      out[[nm]][i] <- metrics[i][[nm]]
    }
  }

  out
}

tabnet_impl <- function(x, y, config = tabnet_config()) {

  has_valid <- config$valid_split > 0

  if (has_valid) {
    n <- nrow(x)
    valid_idx <- sample.int(n, n*config$valid_split)

    if (is.data.frame(y)) {
      valid_y <- y[valid_idx,]
      train_y <- y[-valid_idx,]
    } else if (is.numeric(y) || is.factor(y)) {
      valid_y <- y[valid_idx]
      train_y <- y[-valid_idx]
    }

    valid_data <- list(x = x[valid_idx, ], y = valid_y)
    x <- x[-valid_idx, ]
    y <- train_y
  }

  # training data
  data <- resolve_data(x, y)
  dl <- torch::dataloader(
    torch::tensor_dataset(x = data$x, y = data$y),
    batch_size = config$batch_size,
    drop_last = config$drop_last,
    shuffle = TRUE
  )

  # validation data
  if (has_valid) {
    valid_data <- resolve_data(valid_data$x, valid_data$y)
    valid_dl <- torch::dataloader(
      torch::tensor_dataset(x = valid_data$x, y = valid_data$y),
      batch_size = config$batch_size,
      drop_last = FALSE,
      shuffle = FALSE
    )
  }

  if (config$loss == "auto") {
    if (data$y$dtype == torch::torch_long())
      config$loss <- "cross_entropy"
    else
      config$loss <- "mse"
  }

  # resolve loss
  if (config$loss == "mse")
    config$loss_fn <- torch::nn_mse_loss()
  else if (config$loss %in% c("bce", "cross_entropy"))
    config$loss_fn <- torch::nn_cross_entropy_loss()

  # create network
  network <- tabnet(
    input_dim = data$input_dim,
    output_dim = data$output_dim,
    cat_idxs = data$cat_idx,
    n_d = config$n_d,
    n_a = config$n_a,
    n_steps = config$n_steps,
    gamma = config$gamma,
    virtual_batch_size = config$virtual_batch_size
  )

  # define optimizer
  optimizer <- torch::optim_adam(network$parameters, lr = 2e-2)

  # main loop
  metrics <- list()
  for (epoch in seq_len(config$epochs)) {

    metrics[[epoch]] <- list(train = NULL, valid = NULL)
    train_metrics <- c()
    valid_metrics <- c()

    network$train()

    if (config$verbose)
      pb <- progress::progress_bar$new(
        total = length(dl),
        format = "[:bar] loss= :loss"
      )

    for (batch in torch::enumerate(dl)) {
      m <- train_batch(network, optimizer, batch, config)
      if (config$verbose) pb$tick(tokens = m)
      train_metrics <- c(train_metrics, m)
    }
    metrics[[epoch]][["train"]] <- transpose_metrics(train_metrics)

    network$eval()
    if (has_valid) {
      for (batch in torch::enumerate(valid_dl)) {
        m <- valid_batch(network, batch, config)
        valid_metrics <- c(valid_metrics, m)
      }
      metrics[[epoch]][["valid"]] <- transpose_metrics(valid_metrics)
    }

    message <- sprintf("[Epoch %03d] Loss: %3f", epoch, mean(metrics[[epoch]]$train$loss))
    if (has_valid)
      message <- paste0(message, sprintf(" Valid loss: %3f", mean(metrics[[epoch]]$valid$loss)))

    if (config$verbose)
      rlang::inform(message)
  }

  list(
    network = network,
    metrics = metrics,
    config = config
  )
}

predict_impl <- function(obj, x) {
  data <- resolve_data(x, y = data.frame(rep(1, nrow(x))))

  network <- obj$fit$network
  network$eval()

  network(data$x)[[1]]
}

predict_impl_numeric <- function(obj, x) {
  p <- as.numeric(predict_impl(obj, x))
  hardhat::spruce_numeric(p)
}

get_blueprint_levels <- function(obj) {
  levels(obj$blueprint$ptypes$outcomes[[1]])
}

predict_impl_prob <- function(obj, x) {
  p <- predict_impl(obj, x)
  p <- torch::nnf_softmax(p, dim = 2)
  p <- as.matrix(p)
  hardhat::spruce_prob(get_blueprint_levels(obj), p)
}

predict_impl_class <- function(obj, x) {
  p <- predict_impl(obj, x)
  p <- torch::torch_max(p, dim = 2)
  p <- as.integer(p[[2]])
  p <- get_blueprint_levels(obj)[p]
  p <- factor(p, levels = get_blueprint_levels(obj))
  hardhat::spruce_class(p)
}

