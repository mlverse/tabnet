train_batch_un <- function(network, optimizer, batch, config) {
  # forward pass
  output <- network(batch)
  loss <- config$loss_fn(output[[1]], output[[2]], output[[3]])

  # step of the backward pass and optimization
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

valid_batch_un <- function(network, batch, config) {
  # forward pass
  output <- network(batch)
  loss <- config$loss_fn(output[[1]], output[[2]], output[[3]])

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

unsupervised_loss <- function(y_pred, embedded_x, obfuscation_mask, eps = 1e-9) {

  errors <- y_pred - embedded_x
  reconstruction_errors <- torch::torch_mul(errors, obfuscation_mask)^2
  batch_stds <- torch::torch_std(embedded_x, dim=1)^2 + eps

  # compute the number of obfuscated variables to reconstruct
  nb_reconstructed_variables <- torch::torch_sum(obfuscation_mask, dim=2)

  # take the mean of the reconstructed variable errors
  features_loss <- torch::torch_matmul(reconstruction_errors, 1/batch_stds) / (nb_reconstructed_variables + eps)
  loss <- torch::torch_mean(features_loss)
  loss
}

tabnet_train_unsupervised <- function(x, config = tabnet_config(), epoch_shift = 0L) {
  torch::torch_manual_seed(sample.int(1e6, 1))

  if (config$device == "auto") {
    if (torch::cuda_is_available())
      device <- "cuda"
    else
      device <- "cpu"
  } else {
    device <- config$device
  }

  # dataset to dataloaders
  has_valid <- config$valid_split > 0
  if (has_valid) {
    n <- nrow(x)
    valid_idx <- sample.int(n, n*config$valid_split)
    valid_data <- list(x = x[valid_idx, ])
    x <- x[-valid_idx, ]

  }
  # training data
  data <- resolve_data(x, y=matrix(rep(1, nrow(x)),ncol=1))
  dl <- torch::dataloader(
    torch::tensor_dataset(x = data$x),
    batch_size = config$batch_size,
    drop_last = config$drop_last,
    shuffle = TRUE
  )

  # validation data
  if (has_valid) {
    valid_data <- resolve_data(valid_data$x, y=matrix(rep(1, nrow(valid_data$x)),ncol=1))
    valid_dl <- torch::dataloader(
      torch::tensor_dataset(x = valid_data$x),
      batch_size = config$batch_size,
      drop_last = FALSE,
      shuffle = FALSE
    )
  }

  # resolve loss (shortcutted from config)
  config$loss_fn <- unsupervised_loss

  # create network
  network <- tabnet_pretrainer(
    input_dim = data$input_dim,
    cat_idxs = data$cat_idx,
    cat_dims = data$cat_dims,
    pretraining_ratio = config$pretraining_ratio,
    n_d = config$n_d,
    n_a = config$n_a,
    n_steps = config$n_steps,
    gamma = config$gamma,
    virtual_batch_size = config$virtual_batch_size,
    cat_emb_dim = config$cat_emb_dim,
    n_independent = config$n_independent,
    n_shared = config$n_shared,
    momentum = config$momentum
  )

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

  # initialize metrics & checkpoints
  metrics <- list()
  checkpoints <- list()

  # main loop
  for (epoch in seq_len(config$epochs) + epoch_shift) {

    metrics[[epoch]] <- list(train = NULL, valid = NULL)
    train_metrics <- c()
    valid_metrics <- c()

    network$train()

    if (config$verbose)
      pb <- progress::progress_bar$new(
        total = length(dl),
        format = "[:bar] loss= :loss"
      )

    coro::loop(for (batch in dl) {
      batch_x_to_device <- batch$x$to(device=device)
      m <- train_batch_un(network, optimizer, batch_x_to_device, config)
      if (config$verbose) pb$tick(tokens = m)
      train_metrics <- c(train_metrics, m)
    })
    metrics[[epoch]][["train"]] <- transpose_metrics(train_metrics)

    if (config$checkpoint_epochs > 0 && epoch %% config$checkpoint_epochs == 0) {
      network$to(device = "cpu")
      checkpoints[[length(checkpoints) + 1]] <- model_to_raw(network)
      network$to(device = device)
    }

    network$eval()
    if (has_valid) {
      coro::loop(for (batch in valid_dl) {
        batch_x_to_device <- batch$x$to(device=device)
        m <- valid_batch_un(network, batch_x_to_device, config)
        valid_metrics <- c(valid_metrics, m)
      })
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
