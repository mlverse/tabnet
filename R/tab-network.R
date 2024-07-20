initialize_non_glu <- function(module, input_dim, output_dim) {
  gain_value <- sqrt((input_dim + output_dim)/sqrt(4*input_dim))
  torch::nn_init_xavier_normal_(module$weight, gain = gain_value)
}

initialize_glu <- function(module, input_dim, output_dim) {
  gain_value <- sqrt((input_dim + output_dim)/sqrt(input_dim))
  torch::nn_init_xavier_normal_(module$weight, gain = gain_value)
}

# Ghost Batch Normalization
# https://arxiv.org/abs/1705.08741
#
gbn <- torch::nn_module(
  "gbn",
  initialize = function(input_dim, virtual_batch_size=128, momentum = 0.02) {
    self$input_dim <- input_dim
    self$virtual_batch_size <- virtual_batch_size
    self$bn = torch::nn_batch_norm1d(self$input_dim, momentum = momentum)
  },
  forward = function(x) {
    chunks <- x$chunk(as.integer(ceiling(x$shape[1] / self$virtual_batch_size)), 1)
    res <- lapply(chunks, self$bn)
    torch::torch_cat(res, dim = 1)
  }
)

# Defines main part of the TabNet network without the embedding layers.
#
#
tabnet_encoder <- torch::nn_module(
  "tabnet_encoder",
  initialize = function(input_dim, output_dim,
                        n_d=8, n_a=8,
                        n_steps=3, gamma=1.3,
                        n_independent=2, n_shared=2, epsilon=1e-15,
                        virtual_batch_size=128, momentum = 0.02,
                        mask_type="sparsemax") {

    self$input_dim <- input_dim
    self$output_dim <- output_dim
    self$n_d <- n_d
    self$n_a <- n_a
    self$n_steps <- n_steps
    self$gamma <- gamma
    self$epsilon <- epsilon
    self$n_independent <- n_independent
    self$n_shared <- n_shared
    self$virtual_batch_size <- virtual_batch_size
    self$mask_type <- mask_type
    self$initial_bn <- torch::nn_batch_norm1d(self$input_dim, momentum = momentum)

    if (self$n_shared > 0) {
      shared_feat_transform <- torch::nn_module_list()

      for (i in seq_len(self$n_shared)) {
        if (i == 1) {
          shared_feat_transform$append(torch::nn_linear(
            self$input_dim,
            2*(n_d + n_a),
            bias = FALSE)
          )
        } else {
          shared_feat_transform$append(torch::nn_linear(
            n_d + n_a, 2*(n_d + n_a), bias = FALSE
          ))
        }
      }

    } else {
      shared_feat_transform <- NULL
    }

    self$initial_splitter <- feat_transformer(
      self$input_dim, n_d + n_a, shared_feat_transform,
      n_glu_independent = self$n_independent,
      virtual_batch_size = self$virtual_batch_size,
      momentum = momentum
    )

    self$feat_transformers <- torch::nn_module_list()
    self$att_transformers <- torch::nn_module_list()

    for (step in seq_len(n_steps)) {

      transformer <- feat_transformer(self$input_dim, n_d + n_a, shared_feat_transform,
                                      n_glu_independent = self$n_independent,
                                      virtual_batch_size = self$virtual_batch_size,
                                      momentum = momentum)
      attention <- attentive_transformer(n_a, self$input_dim,
                                         virtual_batch_size = self$virtual_batch_size,
                                         momentum = momentum,
                                         mask_type = self$mask_type)

      self$feat_transformers$append(transformer)
      self$att_transformers$append(attention)

    }

  },
  forward = function(x, prior) {
    res <- torch::torch_tensor(0, device = x$device)
    x <- self$initial_bn(x)

    #prior <- torch::torch_ones(size = x$shape, device = x$device)
    M_loss <- 0
    att <- self$initial_splitter(x)[, (self$n_d + 1):N]

    steps_output <- list()
    for (step in seq_len(self$n_steps)) {

      M <- self$att_transformers[[step]](prior, att)
      M_loss <- M_loss + torch::torch_mean(torch::torch_sum(
        torch::torch_mul(M, torch::torch_log(M + self$epsilon)),
        dim = 2
      ))

      # update prior
      prior <- torch::torch_mul(self$gamma - M, prior)

      # output
      masked_x <- torch::torch_mul(M, x)
      out <- self$feat_transformers[[step]](masked_x)
      d <- torch::nnf_relu(out[.., 1:(self$n_d)])
      steps_output[[step]] <- d
      res <- torch::torch_add(res, d)
      # update attention
      att <- out[, (self$n_d + 1):N]

    }

    M_loss <- M_loss / self$n_steps

    list(res, M_loss, steps_output)
  },
  forward_masks = function(x, x_na_mask) {

    x <- self$initial_bn(x)

    prior <- x_na_mask$logical_not()
    M_explain <- torch::torch_zeros(x$shape, device = x$device)
    att <- self$initial_splitter(x)[, (self$n_d+1):N]
    masks <- list()

    for (step in seq_len(self$n_steps)) {

      M <- self$att_transformers[[step]](prior, att)
      masks[[step]] <- M

      # update prior
      prior <- torch::torch_mul(self$gamma - M, prior)

      # output
      masked_x <- torch::torch_mul(M, x)
      out <- self$feat_transformers[[step]](masked_x)
      d <- torch::nnf_relu(out[.., 1:(self$n_d)])

      # explain
      step_importance <- torch::torch_sum(d, dim=2)
      M_explain <- M_explain + torch::torch_mul(M, step_importance$unsqueeze(dim=2))

      # update attention
      att <- out[, (self$n_d+1):N]

    }

    list(M_explain, masks)
  }
)

tabnet_decoder <- torch::nn_module(
  "tabnet_decoder",
  initialize = function(input_dim, n_d=8,
                        n_steps=3, n_independent=2, n_shared=2,
                        virtual_batch_size = 128, momentum = 0.02) {

    self$input_dim <- input_dim
    self$n_d <- n_d
    self$n_steps <- n_steps
    self$n_independent <- n_independent
    self$n_shared <- n_shared
    self$virtual_batch_size <- virtual_batch_size

    self$feat_transformers <- torch::nn_module_list()
    self$reconstruction_layers <- torch::nn_module_list()

    if (self$n_shared > 0) {
      shared_feat_transform <- torch::nn_module_list()

      for (i in seq_len(self$n_shared)) {
        if (i == 1) {
          shared_feat_transform$append(torch::nn_linear(
            n_d, 2 * n_d, bias = FALSE)
          )
        } else {
          shared_feat_transform$append(torch::nn_linear(
            n_d, 2 * n_d, bias = FALSE)
          )
        }
      }

    } else {
      shared_feat_transform <- NULL
    }

    for (step in seq_len(n_steps)) {

      transformer <- feat_transformer(n_d, n_d, shared_feat_transform,
                                      n_glu_independent=self$n_independent,
                                      virtual_batch_size=self$virtual_batch_size,
                                      momentum = momentum)

      self$feat_transformers$append(transformer)
      reconstruction_layer <- torch::nn_linear(n_d, self$input_dim, bias = FALSE)
      initialize_non_glu(reconstruction_layer, n_d, self$input_dim)
      self$reconstruction_layers$append(reconstruction_layer)

    }

  },
  forward = function(steps_output) {
    res <- torch::torch_tensor(0, device = steps_output[[1]]$device)
    for (step_nb in seq_along(steps_output)) {

      x <- self$feat_transformers[[step_nb]](steps_output[[step_nb]])
      x <- self$reconstruction_layers[[step_nb]](steps_output[[step_nb]])
      res <- torch::torch_add(res, x)

    }

    res
  },
)

tabnet_pretrainer <- torch::nn_module(
  "tabnet_pretrainer",
  initialize = function(input_dim, pretraining_ratio = 0.2,
                        n_d = 8, n_a = 8,
                        n_steps = 3, gamma = 1.3,
                        cat_idxs = c(), cat_dims = c(),
                        cat_emb_dim = 1, n_independent = 2,
                        n_shared = 2, n_independent_decoder = 1,
                        n_shared_decoder = 1, epsilon = 1e-15,
                        virtual_batch_size = 128, momentum = 0.02,
                        mask_type = "sparsemax") {

    self$input_dim <- input_dim
    self$pretraining_ratio <- pretraining_ratio

    # a check par, just to easily find out when we need to
    # reload the model
    self$.check <- torch::nn_parameter(torch::torch_tensor(1, requires_grad = TRUE))
    self$n_d <- n_d
    self$n_a <- n_a
    self$n_steps <- n_steps
    self$gamma <- gamma
    self$cat_idxs <- cat_idxs
    self$cat_dims <- cat_dims
    self$cat_emb_dim <- cat_emb_dim
    self$epsilon <- epsilon
    self$n_independent <- n_independent
    self$n_shared <- n_shared
    self$n_independent_decoder <- n_independent_decoder
    self$n_shared_decoder <- n_shared_decoder
    self$mask_type <- mask_type
    self$initial_bn <- torch::nn_batch_norm1d(self$input_dim, momentum = momentum)

    if (self$n_steps <= 0)
      stop("'n_steps' should be a positive integer.")
    if (self$n_independent == 0 && self$n_shared == 0)
      stop("'n_shared' and 'n_independant' can't be both zero.")

    self$virtual_batch_size <- virtual_batch_size
    self$embedder <- embedding_generator(input_dim, cat_dims, cat_idxs, cat_emb_dim)
    self$embedder_na <- na_embedding_generator(input_dim, cat_dims, cat_idxs, cat_emb_dim)
    self$post_embed_dim <- self$embedder$post_embed_dim
    self$masker = random_obfuscator(self$pretraining_ratio)
    self$encoder = tabnet_encoder(
      input_dim = self$post_embed_dim,
      output_dim = self$post_embed_dim,
      n_d = n_d,
      n_a = n_a,
      n_steps = n_steps,
      gamma = gamma,
      n_independent = n_independent,
      n_shared = n_shared,
      epsilon = epsilon,
      virtual_batch_size = virtual_batch_size,
      momentum = momentum,
      mask_type = mask_type
    )
    self$decoder = tabnet_decoder(
      self$post_embed_dim,
      n_d = n_d,
      n_steps = n_steps,
      n_independent = n_independent_decoder,
      n_shared = n_shared_decoder,
      virtual_batch_size = virtual_batch_size,
      momentum = momentum
    )

  },
  forward = function(x, x_na_mask) {
    embedded_x <- self$embedder(x)
    embedded_x_na_mask <- self$embedder_na(x_na_mask)

    if (self$training) {
      c(masked_x, obfuscated_vars) %<-% self$masker(embedded_x, embedded_x_na_mask)
      # set prior of encoder as !obf_mask
      prior <- obfuscated_vars$logical_not()
      steps_out <- self$encoder(masked_x, prior)[[3]]
      res <- self$decoder(steps_out)
      list(res,
           embedded_x,
           obfuscated_vars)
    } else {
      prior <- embedded_x_na_mask$logical_not()
      steps_out <- self$encoder(embedded_x, prior)[[3]]
      res <- self$decoder(steps_out)
      list(res,
           embedded_x,
           embedded_x_na_mask)
    }
  },
  forward_masks = function(x, x_na_mask) {
    embedded_x <- self$embedder(x)
    embedded_x_na_mask <- self$embedder_na(x_na_mask)
    self$encoder$forward_masks(embedded_x, embedded_x_na_mask)
  }

)

tabnet_no_embedding <- torch::nn_module(
  "tabnet_no_embedding",
  initialize = function(input_dim, output_dim,
                        n_d=8, n_a=8,
                        n_steps=3, gamma=1.3,
                        n_independent=2, n_shared=2, epsilon=1e-15,
                        virtual_batch_size=128, momentum = 0.02,
                        mask_type="sparsemax") {

    self$input_dim <- input_dim
    self$output_dim <- output_dim
    self$is_multi_outcome <- !is.atomic(output_dim)
    self$n_d <- n_d
    self$n_a <- n_a
    self$n_steps <- n_steps
    self$gamma <- gamma
    self$epsilon <- epsilon
    self$n_independent <- n_independent
    self$n_shared <- n_shared
    self$virtual_batch_size <- virtual_batch_size
    self$mask_type <- mask_type
    self$initial_bn <- torch::nn_batch_norm1d(self$input_dim, momentum = momentum)

    self$encoder <- tabnet_encoder(
      input_dim = input_dim,
      output_dim = output_dim,
      n_d = n_d,
      n_a = n_a,
      n_steps = n_steps,
      gamma = gamma,
      n_independent = n_independent,
      n_shared = n_shared,
      epsilon = epsilon,
      virtual_batch_size = virtual_batch_size,
      momentum = momentum,
      mask_type = mask_type
    )
    if (self$is_multi_outcome) {
      self$multi_outcome_mapping <- torch::nn_module_list()
      for (task_dim in output_dim) {
        task_mapping <- torch::nn_linear(n_d, task_dim, bias = FALSE)
        initialize_non_glu(task_mapping, n_d, task_dim)
        self$multi_outcome_mapping$append(task_mapping)
      }
    } else {
      self$final_mapping <- torch::nn_linear(n_d, sum(output_dim), bias = FALSE)
      initialize_non_glu(self$final_mapping, n_d, sum(output_dim))
    }

  },
  forward = function(x, x_na_mask) {
    prior <- x_na_mask$logical_not()
    c(res, M_loss, steps_output) %<-% self$encoder(x, prior)
    if (self$is_multi_outcome) {
      out <- torch::torch_stack(purrr::map(self$multi_outcome_mapping, exec, !!!res), dim = 2)$squeeze(3)
    } else {
      out <- self$final_mapping(res)
    }

    list(out, M_loss)
  },
  forward_masks = function(x, x_na_mask) {
    self$encoder$forward_masks(x, x_na_mask)
    }
)


#' TabNet Model Architecture
#'
#' This is a `nn_module` representing the TabNet architecture from
#' [Attentive Interpretable Tabular Deep Learning](https://arxiv.org/abs/1908.07442).
#'
#' @param input_dim Initial number of features.
#' @param output_dim Dimension of network output. Examples : one for regression, 2 for
#'   binary classification etc.. Vector of those dimensions in case of multi-output.
#' @param n_d Dimension of the prediction layer (usually between 4 and 64).
#' @param n_a Dimension of the attention layer (usually between 4 and 64).
#' @param n_steps Number of successive steps in the network (usually between 3 and 10).
#' @param gamma Scaling factor for attention updates (usually between 1 and 2).
#' @param cat_idxs Index of each categorical column in the dataset.
#' @param cat_dims Number of categories in each categorical column.
#' @param cat_emb_dim Size of the embedding of categorical features if int, all categorical
#'   features will have same embedding size if list of int, every corresponding feature will have
#'   specific size.
#' @param n_independent Number of independent GLU layer in each GLU block of the encoder.
#' @param n_shared Number of shared GLU layer in each GLU block of the encoder.
#' @param epsilon Avoid log(0), this should be kept very low.
#' @param virtual_batch_size Batch size for Ghost Batch Normalization.
#' @param momentum  Numerical value between 0 and 1 which will be used for momentum in all batch norm.
#' @param mask_type Either "sparsemax" or "entmax" : this is the masking function to use.
#' @export
tabnet_nn <- torch::nn_module(
  "tabnet",
  initialize = function(input_dim, output_dim, n_d=8, n_a=8,
                        n_steps=3, gamma=1.3, cat_idxs=c(), cat_dims=c(), cat_emb_dim=1,
                        n_independent=2, n_shared=2, epsilon=1e-15,
                        virtual_batch_size = 128, momentum = 0.02,
                        mask_type="sparsemax") {
    self$cat_idxs <- cat_idxs
    self$cat_dims <- cat_dims
    self$cat_emb_dim <- cat_emb_dim

    # a check par, just to easily find out when we need to
    # reload the model
    self$.check <- torch::nn_parameter(torch::torch_tensor(1, requires_grad = TRUE))

    self$input_dim <- input_dim
    self$output_dim <- output_dim
    self$n_d <- n_d
    self$n_a <- n_a
    self$n_steps <- n_steps
    self$gamma <- gamma
    self$epsilon <- epsilon
    self$n_independent <- n_independent
    self$n_shared <-  n_shared
    self$mask_type <- mask_type

    if (self$n_steps <= 0)
      stop("'n_steps' should be a positive integer.")
    if (self$n_independent == 0 && self$n_shared == 0)
      stop("'n_shared' and 'n_independant' can't be both zero.")

    self$virtual_batch_size <- virtual_batch_size
    self$embedder <- embedding_generator(input_dim, cat_dims, cat_idxs, cat_emb_dim)
    self$embedder_na <- na_embedding_generator(input_dim, cat_dims, cat_idxs, cat_emb_dim)
    self$post_embed_dim <- self$embedder$post_embed_dim
    self$tabnet <- tabnet_no_embedding(self$post_embed_dim, output_dim, n_d, n_a, n_steps,
                                     gamma, n_independent, n_shared, epsilon,
                                     virtual_batch_size, momentum, mask_type)

  },
  forward = function(x, x_na_mask) {
    embedded_x <- self$embedder(x)
    embedded_x_na_mask <- self$embedder_na(x_na_mask)
    self$tabnet(embedded_x, embedded_x_na_mask)
  },
  forward_masks = function(x, x_na_mask) {
    embedded_x <- self$embedder(x)
    embedded_x_na_mask <- self$embedder_na(x_na_mask)
    self$tabnet$forward_masks(embedded_x, embedded_x_na_mask)
  }
)

attentive_transformer <- torch::nn_module(
  "attentive_transformer",
  initialize = function(input_dim, output_dim,
                        virtual_batch_size = 128,
                        momentum = 0.02,
                        mask_type="sparsemax") {
    self$fc <- torch::nn_linear(input_dim, sum(output_dim), bias=FALSE)
    initialize_non_glu(self$fc, input_dim, sum(output_dim))
    self$bn <- gbn(sum(output_dim), virtual_batch_size=virtual_batch_size,
                  momentum = momentum)


    if (mask_type == "sparsemax")
      self$selector <- torch::nn_contrib_sparsemax(dim =-1)
    else if (mask_type == "entmax")
      self$selector <- entmax(dim = -1)
    else
      stop("Please choose either 'sparsemax' or 'entmax' as 'mask_type'")

  },
  forward = function(priors, processed_feat) {
    x <- self$fc(processed_feat)
    x <- self$bn(x)
    x <- torch::torch_mul(x, priors)
    x <- self$selector(x)
    x
  }
)

feat_transformer <- torch::nn_module(
  "feat_transformer",
  initialize = function(input_dim, output_dim, shared_layers, n_glu_independent,
                        virtual_batch_size=128, momentum = 0.02) {

    params <- list(
      'n_glu'= n_glu_independent,
      'virtual_batch_size'= virtual_batch_size,
      'momentum'= momentum
    )

    if (is.null(shared_layers)) {
      self$shared <- torch::nn_identity()
      is_first <- TRUE
    } else {
      self$shared <- glu_block(input_dim, output_dim,
                               first=TRUE,
                               shared_layers=shared_layers,
                               n_glu=length(shared_layers),
                               virtual_batch_size=virtual_batch_size,
                               momentum = momentum)
      is_first <- FALSE
    }

    if (n_glu_independent == 0) {
      self$specifics <- torch::nn_identity()
    } else {
      if (is_first)
        spec_input_dim <- input_dim
      else
        spec_input_dim <- output_dim
      self$specifics <- do.call(glu_block, append(
        list(spec_input_dim, output_dim, first = is_first),
        params
      ))
    }
  },
  forward = function(x) {
    x <- self$shared(x)
    x <- self$specifics(x)
    x
  }
)

glu_block <- torch::nn_module(
  "glu_block",
  initialize = function(input_dim, output_dim, n_glu=2, first=FALSE,
                        shared_layers=NULL,
                        virtual_batch_size=128, momentum = 0.02) {

    self$first <- first
    self$shared_layers <- shared_layers
    self$n_glu <- n_glu
    self$glu_layers <- torch::nn_module_list()

    params = list(
      'virtual_batch_size' = virtual_batch_size,
      'momentum' = momentum
    )

    if (length(shared_layers) > 0)
      fc <- shared_layers[[1]]
    else
      fc <- NULL

    self$glu_layers$append(do.call(
      glu_layer,
      append(list(input_dim, output_dim, fc = fc), params)
    ))

    if (self$n_glu >= 2) {
      for (glu_id in 2:(self$n_glu)) {

        if (length(shared_layers) > 0)
          fc <- shared_layers[[glu_id]]
        else
          fc <- NULL

        self$glu_layers$append(do.call(
          glu_layer,
          append(list(output_dim, output_dim, fc = fc), params)
        ))

      }
    }

  },
  forward = function(x) {
    scale <- torch::torch_sqrt(torch::torch_tensor(0.5, device = x$device))

    if (self$first) {
      x <- self$glu_layers[[1]](x)
      layers_left <- seq_len(self$n_glu)[-1]
    } else {
      layers_left <- seq_len(self$n_glu)
    }

    for (glu_id in layers_left) {
      x <- torch::torch_add(x, self$glu_layers[[glu_id]](x))
      x <- x*scale
    }

    x
  }
)

glu_layer <- torch::nn_module(
  "glu_layer",
  initialize = function(input_dim, output_dim, fc=NULL,
                        virtual_batch_size=128, momentum = 0.02) {
    self$output_dim <- sum(output_dim)

    if (!is.null(fc))
      self$fc <- fc
    else
      self$fc <- torch::nn_linear(input_dim, 2 * self$output_dim, bias = FALSE)

    initialize_glu(self$fc, input_dim, 2 * self$output_dim)

    self$bn <- gbn(2 * self$output_dim, virtual_batch_size = virtual_batch_size,
                  momentum = momentum)
  },
  forward = function(x) {
    x <- self$fc(x)
    x <- self$bn(x)
    out <- torch::torch_mul(
      x[, 1:self$output_dim],
      torch::torch_sigmoid(x[, (self$output_dim + 1):N])
    )
    out
  }
)

embedding_generator <- torch::nn_module(
  "embedding_generator",
  initialize = function(input_dim, cat_dims, cat_idxs, cat_emb_dim) {

    if (length(cat_dims) == 0 || length(cat_idxs) == 0) {
      self$skip_embedding <- TRUE
      self$post_embed_dim <- input_dim
      return(invisible(NULL))
    }

    self$skip_embedding <- FALSE
    self$cat_dims <- cat_dims

    if (length(cat_emb_dim) == 1)
      self$cat_emb_dims <- rep(cat_emb_dim, length(cat_idxs))
    else
      self$cat_emb_dims <- cat_emb_dim

    # check that all embeddings dimensions are provided
    if (length(self$cat_emb_dims) != length(cat_dims)){
      msg = paste0("`cat_emb_dim` length must be 1 or the number of categorical predictors, got length ",length(self$cat_emb_dims)," for ",length(cat_dims)," categorical predictors")
      stop(msg)
    }

    self$post_embed_dim <- as.integer(input_dim + sum(self$cat_emb_dims) - length(self$cat_emb_dims))
    self$embeddings <- torch::nn_module_list()

    # Sort dims by cat_idx
    sorted_idx <- order(cat_idxs)
    cat_dims <- cat_dims[sorted_idx]
    self$cat_emb_dims <- self$cat_emb_dims[sorted_idx]

    for (i in seq_along(cat_dims)) {
      self$embeddings$append(
        torch::nn_embedding(
          cat_dims[i],
          self$cat_emb_dims[i]
        )
      )
    }

    # record continuous indices
    self$continuous_idx <- rep(TRUE, input_dim)
    self$continuous_idx[cat_idxs] <- FALSE

  },
  forward = function(x) {

    if (self$skip_embedding) {
      # no embeddings required
      return(x)
    }

    cols <- list()
    cat_feat_counter <- 1L

    for (i in seq_along(self$continuous_idx)) {

      if (self$continuous_idx[i]) {
        # impute nan with 0s
        cols[[i]] <- x[,i]$nan_to_num(0)$to(dtype = torch::torch_float())$view(c(-1, 1))
      } else {
        # nan mask
        mask <- x[, i]$ge(1)$bitwise_and(x[, i]$le(self$cat_dims[cat_feat_counter]))$to(dtype = torch::torch_long())
        # impute nan with 1s (categorical vars are 1-indexed)
        # cols[[i]] <- self$embeddings[[cat_feat_counter]](x[, i]$nan_to_num(1)$to(dtype = torch::torch_long()))
        # obsolete : requires a masking + imputation to impute all nans with torch 0.11
        imputed_xi <- (x[, i]$mul(mask) + (1 - mask))$nan_to_num(1)
        cols[[i]] <- self$embeddings[[cat_feat_counter]](imputed_xi$to(dtype = torch::torch_long()))
        cat_feat_counter <- cat_feat_counter + 1L
      }

    }

    # concat
    post_embeddings <- torch::torch_cat(cols, dim=2)
    post_embeddings
  }
)

na_embedding_generator <- torch::nn_module(
  "na_embedding_generator",
  initialize = function(input_dim, cat_dims, cat_idxs, cat_emb_dim) {

    if (length(cat_dims) == 0 || length(cat_idxs) == 0) {
      self$skip_embedding <- TRUE
      return(invisible(NULL))
    }

    self$skip_embedding <- FALSE

    if (length(cat_emb_dim) == 1)
      self$cat_emb_dims <- rep(cat_emb_dim, length(cat_idxs))
    else
      self$cat_emb_dims <- cat_emb_dim

    # Sort dims by cat_idx
    sorted_idx <- order(cat_idxs)
    self$cat_emb_dims <- self$cat_emb_dims[sorted_idx]

    # record continuous indices
    self$continuous_idx <- rep(TRUE, input_dim)
    self$continuous_idx[cat_idxs] <- FALSE

  },
  forward = function(x) {

    if (self$skip_embedding) {
      # no embeddings required
      return(x)
    }

    cols <- list()
    cat_feat_counter <- 1

    for (i in seq_along(self$continuous_idx)) {

      if (self$continuous_idx[i]) {
        cols[[i]] <- x[,i]$to(dtype = torch::torch_bool())$view(c(-1, 1))
      } else {
        # extend the vector to match the dimention of the embedding_x via matmul
        # TODO basic tensor broadcasting function could be more efficient and have lower footprint
        cols[[i]] <- x[,i:i]$logical_and(torch::torch_ones(1,self$cat_emb_dims[cat_feat_counter], device = x$device))
        cat_feat_counter <- cat_feat_counter + 1
      }

    }

    # concat
    post_embeddings <- torch::torch_cat(cols, dim=2)
    post_embeddings
  }
)

random_obfuscator <- torch::nn_module(
  "random_obfuscator",
  initialize = function(pretraining_ratio) {

    if (pretraining_ratio <= 0 || pretraining_ratio >= 1) {
      pretraining_ratio <- 0.5
    }

    self$pretraining_ratio <- pretraining_ratio

  },
  forward = function(x, x_na_mask) {
    # workaround while torch_bernoulli is not available in CUDA
    ones <- torch::torch_ones_like(x, device="cpu")
    obfuscated_vars <- torch::torch_bernoulli(self$pretraining_ratio * ones)$to(device=x$device)
    masked_input = torch::torch_mul(obfuscated_vars$logical_or(x_na_mask)$logical_not(), x)

    list(masked_input, obfuscated_vars)

  }
)
