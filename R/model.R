
#' Transforms input data into a list of_tensors and parameters for model input
#'
#' The 3 torch tensors being
#' $x , $x_na_mask, $y
#'  and parameters being
#' cat_idx the vector of x categorical predictor index
#' cat_dims the vector of number of levels of each x categorical predictor
#' input_dim  the number of col in `x`
#' output_dim the `ncol(y)` in case of (multi-outcome) regression or
#'            the `nlevels(y)` in case of classification or
#'            the vector of `nlevels(y)` in case of multi-outcome classification
#'
#' @param x a data frame
#' @param y a response vector
#' @noRd
resolve_data <- function(x, y) {
  cat_idx <- which(sapply(x, is.factor))
  cat_dims <- sapply(cat_idx, function(i) nlevels(x[[i]]))
  # convert factors into integers
  if (length(cat_idx) > 0) {
    x[,cat_idx] <- sapply(cat_idx, function(i) as.integer(x[[i]]))
  } else {
    # prevent empty cat idx
    cat_idx <- 0L
    cat_dims <- 0L
  }
  x_tensor <- torch::torch_tensor(as.matrix(x), dtype = torch::torch_float())
  x_na_mask <- x %>% is.na %>% as.matrix %>% torch::torch_tensor(dtype = torch::torch_bool())

  # convert factors to integers, based on the class of target first column
  # TODO do not assume but assert type-consistency of all y cols
  # and record output_dim
  if (is.factor(y[[1]])) {
    y_tensor <- torch::torch_tensor(sapply(y, function(i) as.integer(i)), dtype = torch::torch_long())
    if (is.atomic(y)) {
      output_dim <- nlevels(y)
    } else {
      output_dim <- sapply(y, function(i) nlevels(i))
    }
  } else {
    y_tensor <- torch::torch_tensor(as.matrix(y), dtype = torch::torch_float())
    output_dim <- ncol(y)
  }

  input_dim <- ncol(x)

  list(x = x_tensor, x_na_mask = x_na_mask, y = y_tensor,
       cat_idx = cat_idx,
       output_dim = output_dim,
       input_dim = input_dim, cat_dims = cat_dims)
}

#' Configuration for TabNet models
#'
#' @param batch_size (int) Number of examples per batch, large batch sizes are
#'   recommended. (default: 1024^2)
#' @param penalty This is the extra sparsity loss coefficient as proposed
#'   in the original paper. The bigger this coefficient is, the sparser your model
#'   will be in terms of feature selection. Depending on the difficulty of your
#'   problem, reducing this value could help (default 1e-3).
#' @param clip_value If a float is given this will clip the gradient at
#'   clip_value. Pass `NULL` to not clip.
#' @param loss (character or function) Loss function for training (default to mse
#'   for regression and cross entropy for classification)
#' @param epochs (int) Number of training epochs.
#' @param drop_last (logical) Whether to drop last batch if not complete during
#'   training
#' @param decision_width (int) Width of the decision prediction layer. Bigger values gives
#'   more capacity to the model with the risk of overfitting. Values typically
#'   range from 8 to 64.
#' @param attention_width (int) Width of the attention embedding for each mask. According to
#'   the paper n_d = n_a is usually a good choice. (default=8)
#' @param num_steps (int) Number of steps in the architecture
#'   (usually between 3 and 10)
#' @param feature_reusage (float) This is the coefficient for feature reusage in the masks.
#'   A value close to 1 will make mask selection least correlated between layers.
#'   Values range from 1.0 to 2.0.
#' @param mask_type (character) Final layer of feature selector in the attentive_transformer
#'   block, either `"sparsemax"` or `"entmax"`.Defaults to `"sparsemax"`.
#' @param virtual_batch_size (int) Size of the mini batches used for
#'   "Ghost Batch Normalization" (default=256^2)
#' @param learn_rate initial learning rate for the optimizer.
#' @param optimizer the optimization method. currently only 'adam' is supported,
#'   you can also pass any torch optimizer function.
#' @param valid_split (`[0, 1)`) The fraction of the dataset used for validation.
#'   (default = 0 means no split)
#' @param num_independent Number of independent Gated Linear Units layers at each step of the encoder.
#'   Usual values range from 1 to 5.
#' @param num_shared Number of shared Gated Linear Units at each step of the encoder. Usual values
#'    at each step of the decoder. range from 1 to 5
#' @param num_independent_decoder For pretraining, number of independent Gated Linear Units layers
#'   Usual values range from 1 to 5.
#' @param num_shared_decoder For pretraining, number of shared Gated Linear Units at each step of the
#'    decoder. Usual values range from 1 to 5.
#' @param verbose (logical) Whether to print progress and loss values during
#'   training.
#' @param lr_scheduler if `NULL`, no learning rate decay is used. If "step"
#'   decays the learning rate by `lr_decay` every `step_size` epochs. If "reduce_on_plateau"
#'   decays the learning rate by `lr_decay` when no improvement after `step_size` epochs.
#'   It can also be a [torch::lr_scheduler] function that only takes the optimizer
#'   as parameter. The `step` method is called once per epoch.
#' @param lr_decay multiplies the initial learning rate by `lr_decay` every
#'   `step_size` epochs. Unused if `lr_scheduler` is a `torch::lr_scheduler`
#'   or `NULL`.
#' @param step_size the learning rate scheduler step size. Unused if
#'   `lr_scheduler` is a `torch::lr_scheduler` or `NULL`.
#' @param cat_emb_dim Size of the embedding of categorical features. If int, all categorical
#'   features will have same embedding size, if list of int, every corresponding feature will have
#'   specific embedding size.
#' @param momentum Momentum for batch normalization, typically ranges from 0.01
#'   to 0.4 (default=0.02)
#' @param pretraining_ratio Ratio of features to mask for reconstruction during
#'   pretraining.  Ranges from 0 to 1 (default=0.5)
#' @param checkpoint_epochs checkpoint model weights and architecture every
#'   `checkpoint_epochs`. (default is 10). This may cause large memory usage.
#'   Use `0` to disable checkpoints.
#' @param device the device to use for training. "cpu" or "cuda". The default ("auto")
#'   uses  to "cuda" if it's available, otherwise uses "cpu".
#' @param importance_sample_size sample of the dataset to compute importance metrics.
#'   If the dataset is larger than 1e5 obs we will use a sample of size 1e5 and
#'   display a warning.
#' @param early_stopping_monitor Metric to monitor for early_stopping. One of "valid_loss", "train_loss" or "auto" (defaults to "auto").
#' @param  early_stopping_tolerance Minimum relative improvement to reset the patience counter.
#'  0.01 for 1% tolerance (default 0)
#' @param early_stopping_patience Number of epochs without improving until stopping training. (default=5)
#' @param num_workers (int, optional): how many subprocesses to use for data
#'   loading. 0 means that the data will be loaded in the main process.
#'   (default: `0`)
#' @param skip_importance if feature importance calculation should be skipped (default: `FALSE`)
#' @return A named list with all hyperparameters of the TabNet implementation.
#'
#' @export
tabnet_config <- function(batch_size = 1024^2,
                          penalty = 1e-3,
                          clip_value = NULL,
                          loss = "auto",
                          epochs = 5,
                          drop_last = FALSE,
                          decision_width = NULL,
                          attention_width = NULL,
                          num_steps = 3,
                          feature_reusage = 1.3,
                          mask_type = "sparsemax",
                          virtual_batch_size = 256^2,
                          valid_split = 0,
                          learn_rate = 2e-2,
                          optimizer = "adam",
                          lr_scheduler = NULL,
                          lr_decay = 0.1,
                          step_size = 30,
                          checkpoint_epochs = 10,
                          cat_emb_dim = 1,
                          num_independent = 2,
                          num_shared = 2,
                          num_independent_decoder = 1,
                          num_shared_decoder = 1,
                          momentum = 0.02,
                          pretraining_ratio = 0.5,
                          verbose = FALSE,
                          device = "auto",
                          importance_sample_size = NULL,
                          early_stopping_monitor = "auto",
                          early_stopping_tolerance = 0,
                          early_stopping_patience = 0L,
                          num_workers=0L,
                          skip_importance = FALSE) {
  if (is.null(decision_width) && is.null(attention_width)) {
    decision_width <- 8 # default is 8
  }

  if (is.null(attention_width))
    attention_width <- decision_width

  if (is.null(decision_width))
    decision_width <- attention_width

  list(
    batch_size = batch_size,
    lambda_sparse = penalty,
    clip_value = clip_value,
    loss = loss,
    epochs = epochs,
    drop_last = drop_last,
    n_d = decision_width,
    n_a = attention_width,
    n_steps = num_steps,
    gamma = feature_reusage,
    mask_type = mask_type,
    virtual_batch_size = virtual_batch_size,
    valid_split = valid_split,
    learn_rate = learn_rate,
    optimizer = optimizer,
    lr_scheduler = lr_scheduler,
    lr_decay = lr_decay,
    step_size = step_size,
    checkpoint_epochs = checkpoint_epochs,
    cat_emb_dim = cat_emb_dim,
    n_independent = num_independent,
    n_shared = num_shared,
    n_independent_decoder = num_independent_decoder,
    n_shared_decoder = num_shared_decoder,
    momentum = momentum,
    pretraining_ratio = pretraining_ratio,
    verbose = verbose,
    device = device,
    importance_sample_size = importance_sample_size,
    early_stopping_monitor = resolve_early_stop_monitor(early_stopping_monitor, valid_split),
    early_stopping_tolerance = early_stopping_tolerance,
    early_stopping_patience = early_stopping_patience,
    early_stopping = !(early_stopping_tolerance == 0 || early_stopping_patience == 0),
    num_workers = num_workers,
    skip_importance = skip_importance
  )
}

get_constr_output <- function(x, R) {
    # MCM of the prediction given the hierarchy constraint expressed in the matrix R """
    c_out <- x$unsqueeze(2)$expand(c(x$shape[1], R$shape[2], R$shape[2]))
    R_batch <- R$expand(c(x$shape[1], R$shape[2], R$shape[2]))
    final_out <- torch::torch_max(R_batch * c_out, dim = 3)
    final_out[[1]]
}

max_constraint_output <- function(output, labels, ancestor) {
  constr_output <-  get_constr_output(output, ancestor)
  train_output <-  get_constr_output(labels * output, ancestor)
  labels$bitwise_not() * constr_output + labels * train_output
}

resolve_loss <- function(config, dtype) {
  loss <- config$loss

  if (is.function(loss))
    loss_fn <- loss
  else if (loss %in% c("mse", "auto") && !dtype == torch::torch_long())
    loss_fn <- torch::nn_mse_loss()
  else if ((loss %in% c("bce", "cross_entropy", "auto") && dtype == torch::torch_long()) || !is.null(config$ancestor_tt))
    # cross entropy loss is required
    loss_fn <- torch::nn_cross_entropy_loss()
  else
    stop(gettextf("`%s` is not a valid loss for outcome of type %s", loss, dtype), call. = FALSE)

  loss_fn
}

resolve_early_stop_monitor <- function(early_stopping_monitor, valid_split) {
  if (early_stopping_monitor %in% c("valid_loss", "auto") && valid_split > 0)
    early_stopping_monitor <- "valid_loss"
  else if (early_stopping_monitor %in% c("train_loss", "auto"))
    early_stopping_monitor <- "train_loss"
  else
    stop(gettextf("%s is not a valid early-stopping metric to monitor with `valid_split` = %s", early_stopping_monitor, valid_split), call. = FALSE)

  early_stopping_monitor
}

train_batch <- function(network, optimizer, batch, config) {
  # NULLing values to avoid a R-CMD Check Note "No visible binding for global variable"
  out <- M_loss <- NULL
  # forward pass
  c(out, M_loss) %<-% network(batch$x, batch$x_na_mask)
  # if target is multi-outcome, loss has to be applied to each label-group
  if (max(batch$output_dim$shape) > 1) {
    # multi-outcome
    outcome_nlevels <- as.numeric(batch$output_dim$to(device="cpu"))
    if (!is.null(config$ancestor_tt)) {
      # hierarchical mandates use of `max_constraint_output`
      loss <- torch::torch_sum(torch::torch_stack(purrr::pmap(
        list(
          torch::torch_split(out, outcome_nlevels, dim = 2),
          torch::torch_split(batch$y, rep(1, length(outcome_nlevels)), dim = 2)
        ),
        ~config$loss_fn(max_constraint_output(.x, .y$squeeze(2), config$ancestor_tt))
      )),
      dim = 1)
    } else {
      # use `resolved_loss`
      loss <- torch::torch_sum(torch::torch_stack(purrr::pmap(
        list(
          torch::torch_split(out, outcome_nlevels, dim = 2),
          torch::torch_split(batch$y, rep(1, length(outcome_nlevels)), dim = 2)
        ),
        ~config$loss_fn(.x, .y$squeeze(2))
      )),
      dim = 1)
    }
  } else {
    if (batch$y$dtype == torch::torch_long()) {
      # classifier needs a squeeze for bce loss
      loss <- config$loss_fn(out, batch$y$squeeze(2))
    } else {
      loss <- config$loss_fn(out, batch$y)
    }
  }
  # Add the overall sparsity loss
  loss <- loss - config$lambda_sparse * M_loss

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
  # NULLing values to avoid a R-CMD Check Note "No visible binding for global variable"
  out <- M_loss <- NULL
  # forward pass
  c(out, M_loss) %<-% network(batch$x, batch$x_na_mask)
  # loss has to be applied to each label-group when output_dim is a vector
  if (max(batch$output_dim$shape) > 1) {
    # multi-outcome
    outcome_nlevels <- as.numeric(batch$output_dim$to(device="cpu"))
    if (!is.null(config$ancestor_tt)) {
      # hierarchical mandates use of `max_constraint_output`
      loss <- torch::torch_sum(torch::torch_stack(purrr::pmap(
        list(
          torch::torch_split(out, outcome_nlevels, dim = 2),
          torch::torch_split(batch$y, rep(1, length(outcome_nlevels)), dim = 2)
        ),
        ~config$loss_fn(max_constraint_output(.x, .y$squeeze(2), config$ancestor_tt))
      )),
      dim = 1)
    } else {
      # use `resolved_loss`
      loss <- torch::torch_sum(torch::torch_stack(purrr::pmap(
        list(
          torch::torch_split(out, outcome_nlevels, dim = 2),
          torch::torch_split(batch$y, rep(1, length(outcome_nlevels)), dim = 2)
        ),
        ~config$loss_fn(.x, .y$squeeze(2))
      )),
      dim = 1)
    }
  } else {
    if (batch$y$dtype == torch::torch_long()) {
      # classifier needs a squeeze for bce loss
      loss <- config$loss_fn(out, batch$y$squeeze(2))
    } else {
      loss <- config$loss_fn(out, batch$y)
    }
  }
  # Add the overall sparsity loss
  loss <- loss - config$lambda_sparse * M_loss

  list(
    loss = loss$item()
  )
}

get_device_from_config <- function(config) {
  if (config$device == "auto") {
    if (torch::cuda_is_available()){
      device <- "cuda"
    } else if (torch::backends_mps_is_available()) {
      device <- "mps"
    } else {
      device <- "cpu"
    }
  } else {
    device <- config$device
  }
  device
}

tabnet_initialize <- function(x, y, config = tabnet_config()) {

  torch::torch_manual_seed(sample.int(1e6, 1))
  has_valid <- config$valid_split > 0

  device <- get_device_from_config(config)

  if (has_valid) {
    n <- nrow(x)
    valid_idx <- sample.int(n, n * config$valid_split)
    valid_x <- x[valid_idx, ]
    valid_y <- y[valid_idx, ]
    train_y <- y[-valid_idx, ]
    valid_ds <-   torch::dataset(
      initialize = function() {},
      .getbatch = function(batch) {resolve_data(valid_x[batch,], valid_y[batch, ])},
      .length = function() {nrow(valid_x)}
    )()
    x <- x[-valid_idx, ]
    y <- train_y
  }

  # training dataset
  train_ds <-   torch::dataset(
    initialize = function() {},
    .getbatch = function(batch) {resolve_data(x[batch, ], y[batch, ])},
    .length = function() {nrow(x)}
  )()
  # we can get training_set parameters from the 2 first samples
  train <- train_ds$.getbatch(batch = c(1:2))

  # resolve loss
  config$loss_fn <- resolve_loss(config, train$y$dtype)

  # create network
  network <- tabnet_nn(
    input_dim = train$input_dim,
    output_dim = train$output_dim,
    cat_idxs = train$cat_idx,
    cat_dims = train$cat_dims,
    n_d = config$n_d,
    n_a = config$n_a,
    n_steps = config$n_steps,
    gamma = config$gamma,
    virtual_batch_size = config$virtual_batch_size,
    cat_emb_dim = config$cat_emb_dim,
    n_independent = config$n_independent,
    n_shared = config$n_shared,
    momentum = config$momentum,
    mask_type = config$mask_type
  )

  # main loop
  metrics <- list()
  checkpoints <- list()


  importances <- tibble::tibble(
    variables = colnames(x),
    importance = NA
  )

  list(
    network = network,
    metrics = metrics,
    config = config,
    checkpoints = checkpoints,
    importances = importances
  )
}

tabnet_train_supervised <- function(obj, x, y, config = tabnet_config(), epoch_shift = 0L) {
  stopifnot("tabnet_model shall be initialised or pretrained" = (length(obj$fit$network) > 0))
  torch::torch_manual_seed(sample.int(1e6, 1))

  device <- get_device_from_config(config)

  # validation dataset & dataloaders
  has_valid <- config$valid_split > 0
  if (has_valid) {
    n <- nrow(x)
    valid_idx <- sample.int(n, n * config$valid_split)
    valid_x <- x[valid_idx, ]
    valid_y <- y[valid_idx, ]
    train_y <- y[-valid_idx, ]

    valid_ds <-   torch::dataset(
      initialize = function() {},
      .getbatch = function(batch) {resolve_data(valid_x[batch,], valid_y[batch,])},
      .length = function() {nrow(valid_x)}
    )()

    valid_dl <- torch::dataloader(
      valid_ds,
      batch_size = config$batch_size,
      shuffle = FALSE,
      num_workers = config$num_workers
    )

    x <- x[-valid_idx, ]
    y <- train_y
  }

  # training dataset & dataloader
  train_ds <-   torch::dataset(
    initialize = function() {},
    .getbatch = function(batch) {resolve_data(x[batch,], y[batch,])},
    .length = function() {nrow(x)}
  )()

  train_dl <- torch::dataloader(
    train_ds,
    batch_size = config$batch_size,
    drop_last = config$drop_last,
    shuffle = TRUE ,
    num_workers = config$num_workers
  )

  # resolve loss
  config$loss_fn <- resolve_loss(config, train_ds$.getbatch(batch = c(1:2))$y$dtype)

  # restore network from model and send it to device
  network <- obj$fit$network

  network$to(device = device)

  # provide ancestor to torch tensor in case of hierarchical classification
  if (!is.null(config$ancestor)) {
    config$ancestor_tt <- torch::torch_tensor(config$ancestor)$to(torch::torch_bool(), device = device)
  }
  # define optimizer
  if (rlang::is_function(config$optimizer)) {

    optimizer <- config$optimizer(network$parameters, config$learn_rate)

  } else if (rlang::is_scalar_character(config$optimizer)) {

    if (config$optimizer == "adam")
      optimizer <- torch::optim_adam(network$parameters, lr = config$learn_rate)
    else
      stop("Currently only the 'adam' optimizer is supported.", call. = FALSE)

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
    stop("Currently only the 'step' and 'reduce_on_plateau' scheduler are supported.", call. = FALSE)
  }

  # restore previous metrics & checkpoints
  metrics <- obj$fit$metrics
  checkpoints <- obj$fit$checkpoints
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
      m <- train_batch(network, optimizer, to_device(batch, device), config)
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
        m <- valid_batch(network, to_device(batch, device), config)
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
      if (change > config$early_stopping_tolerance){
        patience_counter <- patience_counter + 1
        if (patience_counter >= config$early_stopping_patience){
          if (config$verbose)
            message(gettextf("Early stopping at epoch %03d", epoch))
          break
        }
      } else {
        # reset the patience counter
        best_metric <- current_loss
        patience_counter <- 0L
      }
    }
    if (config$early_stopping && epoch == 1+epoch_shift) {
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
      warning(
        gettextf(
          "Computing importances for a dataset with size %s. This can consume too much memory. We are going to use a sample of size 1e5, You can disable this message by using the `importance_sample_size` argument.",
          train_ds$.length()))
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

predict_impl <- function(obj, x, batch_size = 1e5) {
  # prediction dataset
  device = obj$fit$config$device
  predict_ds <-   torch::dataset(
    initialize = function() {},
    .getbatch = function(batch) {resolve_data(x[batch,], rep(1, nrow(x)))},
    .length = function() {nrow(x)}
  )()

  network <- obj$fit$network
  num_workers <- obj$fit$config$num_workers
  yhat <- c()
  network$eval()

  predict_dl <- torch::dataloader(
    predict_ds,
    batch_size = batch_size,
    drop_last = FALSE,
    shuffle = FALSE ,
    # num_workers = num_workers
    num_workers = 0L
  )
  coro::loop(for (batch in predict_dl) {
    batch <- to_device(batch, device)
    yhat <- c(yhat, network(batch$x, batch$x_na_mask)[[1]])
  })
  # bind rows of the batches
  torch::torch_cat(yhat)
}

predict_impl_numeric <- function(obj, x, batch_size) {
  p <- as.matrix(predict_impl(obj, x, batch_size))
  hardhat::spruce_numeric(as.numeric(p))
}

predict_impl_numeric_multiple <- function(obj, x, batch_size) {
  p <- as.matrix(predict_impl(obj, x, batch_size))
  # TODO use a cleaner function to turn matrix into vectors
  hardhat::spruce_numeric_multiple(!!!purrr::map(1:ncol(p), ~p[,.x]))
}

#' single-outcome level blueprint
#'
#' @param obj : a tabnet object
#'
#' @return : outcome levels
#' @noRd
get_blueprint_levels <- function(obj) {
  levels(obj$blueprint$ptypes$outcomes[[1]])
}

#' multi-outcome levels blueprint
#'
#' @param obj : a tabnet object
#'
#' @return : a list of levels vectors for each outcome
#' @noRd
get_blueprint_levels_multiple <- function(obj) {
  purrr::map(obj$blueprint$ptypes$outcomes, levels) %>%
    rlang::set_names(names(obj$blueprint$ptypes$outcomes))
}

predict_impl_prob <- function(obj, x, batch_size) {
  p <- predict_impl(obj, x, batch_size)
  p <- torch::nnf_softmax(p, dim = 2)
  p <- as.matrix(p)
  hardhat::spruce_prob(get_blueprint_levels(obj), p)
}

predict_impl_prob_multiple <- function(obj, x, batch_size, outcome_nlevels) {
  p <- predict_impl(obj, x, batch_size)
  p <- torch::nnf_softmax(p, dim = 2)
  p <- as.matrix(p)
  # TODO use a cleaner function to turn matrix into vectors
  p_blueprint <- get_blueprint_levels_multiple(obj)
  p_probs <- purrr::map(torch::torch_split(p, outcome_nlevels, dim = 2),
                        as.matrix)
  hardhat::spruce_prob_multiple(!!!purrr::pmap(
    list(p_blueprint, p_probs),
    # TODO BUG each element of `...` must be a tibble, not a list.
    ~hardhat::spruce_prob(.x, .y)) %>%
      rlang::set_names(names(p_blueprint))
    )
}

predict_impl_class <- function(obj, x, batch_size) {
  p <- predict_impl(obj, x, batch_size)
  p_idx <- as.integer(torch::torch_max(p, dim = 2)[[2]])
  p_idx <- get_blueprint_levels(obj)[p_idx]
  p <- factor(p_idx, levels = get_blueprint_levels(obj))
  hardhat::spruce_class(p)
}

predict_impl_class_multiple <- function(obj, x, batch_size, outcome_nlevels) {
  p <- predict_impl(obj, x, batch_size)
  p_levels <- get_blueprint_levels_multiple(obj)
  p_idx <- purrr::map(
    torch::torch_split(p, outcome_nlevels, dim = 2),
    ~as.integer(torch::torch_max(.x, dim = 2)[[2]])
    ) %>% rlang::set_names(names(p_levels))
  p_factor_lst <- purrr::pmap(
    list(p_idx, p_levels),
    ~factor(.y[.x], levels = .y)
  )
  hardhat::spruce_class_multiple(!!!p_factor_lst)
}

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
