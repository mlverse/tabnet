
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

tabnet_config <- function(...) {
  list(
    batch_size = 256,
    lambda_sparse = 1e-3,
    clip_value = 1,
    loss = "mse",
    epochs = 1000,
    drop_last = FALSE,
    n_d = 8,
    n_a = 8,
    n_steps = 3,
    gamma = 1.3,
    virtual_batch_size = 128
  )
}

train_batch <- function(batch, config) {
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

valid_batch <- function(batch, config) {
  # forward pass
  output <- network(batch$x)
  loss <- config$loss_fn(output[[1]], batch$y)

  # Add the overall sparsity loss
  loss <- loss - config$lambda_sparse * output[[2]]

  list(
    loss = loss$item()
  )
}

fit_tabnet <- function(x, y, valid_data = NULL, config = tabnet_config()) {

  # training data
  data <- resolve_data(x, y)
  dl <- torch::dataloader(
    torch::tensor_dataset(x = data$x, y = data$y),
    batch_size = config$batch_size,
    drop_last = config$drop_last,
    shuffle = TRUE
  )

  # validation data
  has_valid <- TRUE
  if (is.null(valid_data)) {
    valid_data <- resolve_data(valid_data$x, valid_data$y)
    valid_dl <- torch::dataloader(
      torch::tensor_dataset(x = valid_data$x, y = valid_data$y),
      batch_size = config$batch_size,
      drop_last = FALSE,
      shuffle = FALSE
    )
    has_valid <- TRUE
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

    metrics[[epoch]] <- list()
    train_metrics <- c()
    valid_metrics <- c()

    network$train()
    for (batch in torch::enumerate(dl)) {
      loss <- train_batch(batch)
      losses <- c(losses, loss)
    }

    network$eval()
    if (has_valid) {
      for (batch in torch::enumerate(valid_dl)) {
        metrics <- valid_batch(batch)
        valid_metrics <- c(valid_metrics, metrics)
      }
      metrics[[epoch]][["valid"]] <- valid_metrics
    }

    metrics[[epoch]][["train"]] <- train_metrics

    message(sprintf("[Epoch %03d] Loss: %3f", epoch, sqrt(mean(losses))))
  }

  list(
    network = network,
    metrics = metrics,
    config = config
  )
}

test <- function() {
  data("ames", package = "modeldata")
  x <- dplyr::select(ames, -Sale_Price)
  y <- ames$Sale_Price
  fit_tabnet(x, y)
}

test_py <- function() {
  data("ames", package = "modeldata")
  x <- dplyr::select(ames, -Sale_Price)
  y <- ames$Sale_Price


  x_ <- x
  for (v in seq_along(x_)) {
    if (is.factor(x_[[v]]))
      x_[[v]] <- as.numeric(x_[[v]])
  }

  cat_idx <- which(sapply(x, is.factor)) - 1
  input_dim <- ncol(x)

  tabnetpy <- reticulate::import("pytorch_tabnet.tab_model")
  reg <- tabnetpy$TabNetRegressor(input_dim = input_dim, output_dim = 1, cat_idxs = cat_idx)
  model <- reg$fit(as.matrix(x_), matrix(y, ncol = 1), max_epochs = 1000L, batch_size = 256L)

}


