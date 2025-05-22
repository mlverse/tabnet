train_batch_un <- function(network, optimizer, batch, config) {
  # forward pass
  output <- network(batch$x, batch$x_na_mask)
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
  output <- network(batch$x, batch$x_na_mask)
  # we inverse the batch_na_mask here to avoid nan in the loss
  loss <- config$loss_fn(output[[1]], output[[2]], output[[3]]$logical_not())

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

  out[-1]
}


tabnet_train_unsupervised <- function(x, config = tabnet_config(), epoch_shift = 0L) {
  torch::torch_manual_seed(sample.int(1e6, 1))

  device <- get_device_from_config(config)

  # validation dataset & dataloaders
  has_valid <- config$valid_split > 0
  if (has_valid) {
    n <- nrow(x)
    valid_idx <- sample.int(n, n*config$valid_split)
    valid_x <- x[valid_idx, ]
    valid_ds <-   torch::dataset(
      initialize = function() {},
      .getbatch = function(batch) {resolve_data(valid_x[batch,], matrix(1L, nrow = length(batch)))},
      .length = function() {nrow(valid_x)}
    )()

    valid_dl <- torch::dataloader(
      valid_ds,
      batch_size = config$batch_size,
      shuffle = FALSE ,
      num_workers = config$num_workers
    )

    x <- x[-valid_idx, ]
  }

  # training dataset & dataloader
  train_ds <-   torch::dataset(
    initialize = function() {},
    .getbatch = function(batch) {resolve_data(x[batch,], matrix(1L, nrow = length(batch)))},
    .length = function() {nrow(x)}
  )()
  # we can get training_set parameters from the 2 first samples
  train <- train_ds$.getbatch(batch = c(1:2))

  train_dl <- torch::dataloader(
    train_ds,
    batch_size = config$batch_size,
    drop_last = config$drop_last,
    shuffle = TRUE ,
    num_workers = config$num_workers
  )

  # resolve loss (shortcutted from config)
  config$loss_fn <- nn_unsupervised_loss()

  # create network
  network <- tabnet_pretrainer(
    input_dim = train$input_dim,
    cat_idxs = train$cat_idx,
    cat_dims = train$cat_dims,
    pretraining_ratio = config$pretraining_ratio,
    n_d = config$n_d,
    n_a = config$n_a,
    n_steps = config$n_steps,
    gamma = config$gamma,
    virtual_batch_size = config$virtual_batch_size,
    cat_emb_dim = config$cat_emb_dim,
    n_independent = config$n_independent,
    n_shared = config$n_shared,
    n_independent_decoder = config$n_independent_decoder,
    n_shared_decoder = config$n_shared_decoder,
    momentum = config$momentum,
    mask_type = config$mask_type,
    mask_topk = config$mask_topk
  )

  network$to(device = device)

  # instantiate optimizer
  if (is_optim_generator(config$optimizer)) {
    optimizer <- config$optimizer(network$parameters, config$learn_rate)
  } else {
    type_error("{.var optimizer} must be resolved into a torch optimizer generator.")
  }
  

  # define scheduler
  if (is.null(config$lr_scheduler)) {
    scheduler <- list(step = function() {})
  } else if (rlang::is_function(config$lr_scheduler)) {
    scheduler <- config$lr_scheduler(optimizer)
  } else if (config$lr_scheduler == "reduce_on_plateau") {
    scheduler <- torch::lr_reduce_on_plateau(optimizer, factor = config$lr_decay, patience = config$step_size)
  } else if (config$lr_scheduler == "step") {
    scheduler <- torch::lr_step(optimizer, config$step_size, config$lr_decay)
  } else {
    not_implemented_error("Currently only the {.str step} and {.str reduce_on_plateau} scheduler are supported.", call. = FALSE)
  }

  # initialize metrics & checkpoints
  metrics <- list()
  checkpoints <- list()
  patience_counter <- 0L

  # main loop
  for (epoch in seq_len(config$epochs) + epoch_shift) {

    metrics[[epoch]] <- list()
    train_metrics <- c()
    valid_metrics <- c()

    network$train()

    if (config$verbose)
      pb <- progress::progress_bar$new(
        total = length(train_dl),
        format = "[:bar] loss= :loss"
      )

    coro::loop(for (batch in train_dl) {
      m <- train_batch_un(network, optimizer, to_device(batch, device), config)
      if (config$verbose) pb$tick(tokens = m)
      train_metrics <- c(train_metrics, m)
    })
    metrics[[epoch]][["train"]] <- transpose_metrics(train_metrics)$loss

    if (config$checkpoint_epochs > 0 && epoch %% config$checkpoint_epochs == 0) {
      network$to(device = "cpu")
      checkpoints[[length(checkpoints) + 1]] <- model_to_raw(network)
      metrics[[epoch]][["checkpoint"]] <- TRUE
      network$to(device = device)
    }

    network$eval()
    if (has_valid) {
      coro::loop(for (batch in valid_dl) {
        m <- valid_batch_un(network, to_device(batch, device), config)
        valid_metrics <- c(valid_metrics, m)
      })
      metrics[[epoch]][["valid"]] <- transpose_metrics(valid_metrics)$loss
    }

    if (config$verbose & !has_valid)
      message(gettextf("[Epoch %03d] Loss: %3f", epoch, mean(metrics[[epoch]]$train)))
    if (config$verbose & has_valid)
      message(gettextf("[Epoch %03d] Loss: %3f, Valid loss: %3f", epoch, mean(metrics[[epoch]]$train), mean(metrics[[epoch]]$valid)))

    # Early-stopping checks
    if (config$early_stopping && config$early_stopping_monitor=="valid_loss"){
      current_loss <- mean(metrics[[epoch]]$valid)
    } else {
      current_loss <- mean(metrics[[epoch]]$train)
    }
    if (config$early_stopping && epoch > 1+epoch_shift) {
      # compute relative change, and compare to best_metric
      change <- (current_loss - best_metric) / current_loss
      if (change > config$early_stopping_tolerance) {
        patience_counter <- patience_counter + 1
        if (patience_counter >= config$early_stopping_patience) {
          if (config$verbose)
            cli::cli_alert_success(gettextf("Early-stopping at epoch {.val epoch}"))
          break
        }
      } else {
        # reset the patience counter
        best_metric <- current_loss
        patience_counter <- 0L
      }
    }
    if (config$early_stopping && epoch == 1 + epoch_shift) {
      # initialise best_metric
      best_metric <- current_loss
    }

    if ("metrics" %in% names(formals(scheduler$step))) {
      scheduler$step(current_loss)
    } else {
      scheduler$step()
    }
  }

  network$to(device = "cpu")
  if(!config$skip_importance) {
    importance_sample_size <- config$importance_sample_size
    if (is.null(config$importance_sample_size) && train_ds$.length() > 1e5) {
      warn("Computing importances for a dataset with size {.val {train_ds$.length()}}. 
           This can consume too much memory. We are going to use a sample of size 1e5. 
           You can disable this message by using the `importance_sample_size` argument.")
      importance_sample_size <- 1e5
    }
    indexes <- as.numeric(torch::torch_randint(
      1, train_ds$.length(), min(importance_sample_size, train_ds$.length()),
      dtype = torch::torch_long()
    ))
    importances <- tibble::tibble(
      variables = colnames(x),
      importance = compute_feature_importance(
        network,
        train_ds$.getbatch(batch =indexes)$x$to(device = "cpu"),
        train_ds$.getbatch(batch =indexes)$x_na_mask$to(device = "cpu"))
    )
  } else {
    importances <- NULL
  }

  list(
    network = network,
    metrics = metrics,
    config = config,
    checkpoints = checkpoints,
    importances = importances
  )
}
