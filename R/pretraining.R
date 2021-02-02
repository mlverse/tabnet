
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

unsupervised_loss <- function(y_pred, embedded_x, obf_vars, eps=1e-9) {
  # TODO current loss functions in train_batch and valid_batch only receive two params : (output[[1]], batch$y)
  errors <- y_pred - embedded_x
  reconstruction_errors <- torch::mul(errors, obf_vars)**2
  batch_stds <- torch::std(embedded_x, dim=0)**2 + eps
  features_loss <- torch::matmul(reconstruction_errors, 1 / batch_stds)
  # here we take the mean per batch, contrary to the paper
  loss <- torch::mean(features_loss)
  loss
}


tabnet_train_unsupervised <- function(obj, x, y, config = tabnet_config(), epoch_shift=OL) {
  stopifnot("tabnet_model shall be initialised or pretrained"= (length(obj$fit$network) > 0))
  torch::torch_manual_seed(sample.int(1e6, 1))
  has_valid <- config$valid_split > 0

  if (config$device == "auto") {
    if (torch::cuda_is_available())
      device <- "cuda"
    else
      device <- "cpu"
  } else {
    device <- config$device
  }

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
  if (config$loss == "mse") {
    config$loss_fn <- torch::nn_mse_loss()
    }  else if (config$loss %in% c("bce", "cross_entropy")) {
      config$loss_fn <- torch::nn_cross_entropy_loss()
    }
  # restore network from model and send it to device
  network <- obj$fit$network

  network$to(device = device)

  # define optimizer

  if (rlang::is_function(config$optimizer)) {

    optimizer <- config$optimizer(network$parameters, config$learn_rate)

  } else if (rlang::is_scalar_character(config$optimizer)) {

    if (config$optimizer == "adam")
      optimizer <- torch::optim_adam(network$parameters, lr = config$learn_rate)
    else
      rlang::abort("Currently only the 'adam' optimizer is supported.")

  }

  # define scheduler

  if (is.null(config$lr_scheduler)) {
    scheduler <- list(step = function() {})
  } else if (rlang::is_function(config$lr_scheduler)) {
    scheduler <- config$lr_scheduler(optimizer)
  } else if (config$lr_scheduler == "step") {
    scheduler <- torch::lr_step(optimizer, config$step_size, config$lr_decay)
  }

  # main loop
  metrics <- obj$fit$metrics
  checkpoints <- obj$fit$checkpoints

  for (epoch in seq_len(config$epochs)+epoch_shift) {

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
      m <- train_batch(network, optimizer, batch_to_device(batch, device), config)
      if (config$verbose) pb$tick(tokens = m)
      train_metrics <- c(train_metrics, m)
    }
    metrics[[epoch]][["train"]] <- transpose_metrics(train_metrics)

    if (config$checkpoint_epochs > 0 && epoch %% config$checkpoint_epochs == 0) {
      network$to(device = "cpu")
      checkpoints[[length(checkpoints) + 1]] <- model_to_raw(network)
      network$to(device = device)
    }

    network$eval()
    if (has_valid) {
      for (batch in torch::enumerate(valid_dl)) {
        m <- valid_batch(network, batch_to_device(batch, device), config)
        valid_metrics <- c(valid_metrics, m)
      }
      metrics[[epoch]][["valid"]] <- transpose_metrics(valid_metrics)
    }

    message <- sprintf("[Epoch %03d] Loss: %3f", epoch, mean(metrics[[epoch]]$train$loss))
    if (has_valid)
      message <- paste0(message, sprintf(" Valid loss: %3f", mean(metrics[[epoch]]$valid$loss)))

    if (config$verbose)
      rlang::inform(message)

    scheduler$step()
  }

  network$to(device = "cpu")

  importances <- tibble::tibble(
    variables = colnames(x),
    importance = compute_feature_importance(network, data$x)
  )

  list(
    network = network,
    metrics = metrics,
    config = config,
    checkpoints = checkpoints,
    importances = importances
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
