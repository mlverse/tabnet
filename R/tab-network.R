initialize_non_glu <- function(module, input_dim, output_dim) {
  gain_value <- sqrt((input_dim + output_dim)/sqrt(4*input_dim))
  torch::nn_init_xavier_normal_(module$weight, gain = gain_value)
}

initialize_glu <- function(module, input_dim, output_dim) {
  gain_value <- sqrt((input_dim + output_dim)/sqrt(input_dim))
  torch::nn_init_xavier_normal_(module$weight, gain = gain_value)
}

#' Ghost Batch Normalization
#' https://arxiv.org/abs/1705.08741
#'
gbn <- torch::nn_module(
  "gbn",
  initialize = function(input_dim, virtual_batch_size=128, momentum=0.01) {
    self$input_dim <- input_dim
    self$virtual_batch_size <- virtual_batch_size
    self$bn = torch::nn_batch_norm1d(self$input_dim, momentum=momentum)
  },
  forward = function(x) {
    chunks <- x$chunk(as.integer(ceiling(x$shape[1] / self$virtual_batch_size)), 1)
    res <- lapply(chunks, self$bn)
    torch::torch_cat(res, dim=1)
  }
)

#' Defines main part of the TabNet network without the embedding layers.
#'
#'
tabnet_no_embedding <- torch::nn_module(
  "tabnet_no_embedding",
  initialize = function(input_dim, output_dim,
                        n_d=8, n_a=8,
                        n_steps=3, gamma=1.3,
                        n_independent=2, n_shared=2, epsilon=1e-15,
                        virtual_batch_size=128, momentum=0.02,
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
    self$initial_bn = torch::nn_batch_norm1d(self$input_dim, momentum=0.01)

    if (self$n_shared > 0) {
      shared_feat_transform <- torch::nn_module_list()

      for (i in seq_along(self$n_shared)) {
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
      self$input_dim, n_d+n_a, shared_feat_transform,
      n_glu_independent=self$n_independent,
      virtual_batch_size=self$virtual_batch_size,
      momentum=momentum
    )

    self$feat_transformers <- torch::nn_module_list()
    self$att_transformers <- torch::nn_module_list()

    for (step in seq_along(n_steps)) {

      transformer <- feat_transformer(self$input_dim, n_d+n_a, shared_feat_transform,
                                      n_glu_independent=self$n_independent,
                                      virtual_batch_size=self$virtual_batch_size,
                                      momentum=momentum)
      attention <- attentive_transformer(n_a, self$input_dim,
                                         virtual_batch_size=self$virtual_batch_size,
                                         momentum=momentum,
                                         mask_type=self$mask_type)

      self$feat_transformers$append(transformer)
      self$att_transformers$append(attention)

    }

    self$final_mapping <- torch::nn_liner(n_d, output_dim, bias=FALSE)
    initialize_non_glu(self$final_mapping, n_d, output_dim)

  },
  forward = function(x) {
    res <- 0
    x <- self$initial_bn(x)

    prior <- torch::torch_ones(size = x$shape, device = x$device())
    M_loss <- 0
    att <- self$initial_splitter(x)[, self$n_d:N]

    for (step in seq_along(self$n_steps)) {
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
      d <- torch::nnf_relu(out[,, 1:self$n_d])
      res <- torch::torch_add(res, d)
      # update attention
      att <- out[, self$n_d:N]

    }


    M_loss <- M_loss/self$n_steps
    res <- self$final_mapping(res)

    list(res, M_loss)
  },
  forward_masks = function(x) {

    x <- self$initial_bn(x)

    prior <- torch::torch_ones(x$shape, device = x$device())
    M_explain <- torch::torch_zeros(x$shape, device = x$device())
    att <- self$initial_splitter(x)[, self$n_d:N]
    masks <- list()

    for (step in seq_along(self$n_steps)) {

      M <- self$att_transformers[[step]](prior, att)
      masks[[step]] <- M

      # update prior
      prior <- torch::torch.mul(self$gamma - M, prior)
      # output
      masked_x <- torch::torch_mul(M, x)
      out <- self$feat_transformers[[step]](masked_x)
      d <- torch::nnf_relu(out[, 1:self$n_d])
      # explain
      step_importance <- torch::torch_sum(d, dim=2)
      M_explain <- M_explain + torch::torch_mul(M, step_importance$unsqueeze(dim=2))
      # update attention
      att <- out[, self$n_d:N]

    }

    list(M_explain, masks)
  }
)

tabnet <- torch::nn_module(
  "tabnet",
  initialize = function(input_dim, output_dim, n_d=8, n_a=8,
                        n_steps=3, gamma=1.3, cat_idxs=c(), cat_dims=c(), cat_emb_dim=1,
                        n_independent=2, n_shared=2, epsilon=1e-15,
                        virtual_batch_size=128, momentum=0.02, device_name='auto',
                        mask_type="sparsemax") {
    self$cat_idxs <- cat_idxs
    self$cat_dims <- cat_dims
    self$cat_emb_dim <- cat_emb_dim

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
      stop("n_steps should be a positive integer.")
    if (self$n_independent == 0 && self$n_shared == 0)
      stop("n_shared and n_independant can't be both zero.")

    self$virtual_batch_size <- virtual_batch_size
    self$embedder <- embedding_generator(input_dim, cat_dims, cat_idxs, cat_emb_dim)
    self$post_embed_dim <- self$embedder$post_embed_dim
    self$tabnet <- tabnet_no_embedding(self$post_embed_dim, output_dim, n_d, n_a, n_steps,
                                     gamma, n_independent, n_shared, epsilon,
                                     virtual_batch_size, momentum, mask_type)

    # Defining device
    if (device_name == 'auto') {
      if (torch::cuda_is_available())
        device_name <- "cuda"
      else
        device_name <- "cpu"
    }
    self$device <- torch::torch_device(device_name)
    self$to(device = self$device)

  },
  forward = function(x) {
    x <- self$embedder(x)
    self$tabnet(x)
  },
  forward_masks = function(x) {
    x <- self$embedder(x)
    self$tabnet$forward_masks(x)
  }
)

attentive_transformer <- torch::nn_module(
  "attentive_transformer",
  initialize = function(input_dim, output_dim,
                        virtual_batch_size=128,
                        momentum=0.02,
                        mask_type="sparsemax") {
    self$fc <- torch::nn_linear(input_dim, output_dim, bias=FALSE)
    initialize_non_glu(self$fc, input_dim, output_dim)
    self$bn <- gbn(output_dim, virtual_batch_size=virtual_batch_size,
                  momentum=momentum)

    if (mask_type == "sparsemax")
      self$selector <- Sparsemax(dim=-1)
    else if (mask_type == "entmax")
      self$selector <- Entmax15(dim=-1)
    else
      stop("Please choose either sparsemax or entmax as masktype")

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
                        virtual_batch_size=128, momentum=0.02) {

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
                               momentum=momentum)
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
