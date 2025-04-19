
# credits to Yandex https://github.com/Qwicen/node/blob/master/lib/nn_utils.py
.make_ix_like <- function(input, dim = 1) {
  d <- input$size(dim)
  rho <- torch::torch_arange(start = 1, end = d, device = input$device,
                             dtype = input$dtype)
  view <- rep(1, input$dim())
  view[1] <- -1
  rho$view(view)$transpose(1, dim)
}

.roll_last <- function(input, dim = 1) {
  if (dim == -1) {
    return(input)
  } else if (dim < 0) {
    dim <- input$dim() - dim
  }
  perm <- c(which(1:input$dim() != dim), dim)
  return(input$permute(perm))
}

.threshold_and_support <- function(input, dim) {
  sorted_input <- input$sort(dim = dim, descending = TRUE)[[1]]
  input_cumsum <- sorted_input$cumsum(dim) - 1
  rhos <- .make_ix_like(input, dim)
  support <- rhos * sorted_input > input_cumsum

  support_size <- support$sum(dim=dim)$unsqueeze(dim)
  tau <- input_cumsum$gather(dim, support_size)
  tau$div_(support_size$to(dtype = input$dtype))
  list(
    tau,
    support_size
  )
}

sparsemax_function <- torch::autograd_function(
  forward = function(ctx, input, dim = -1) {
    max_val <- input$max(dim=dim, keepdim=TRUE)[[1]]
    input$sub_(max_val) # same numerical stability trick as for softmax
    tau_supp_size = .threshold_and_support(input, dim=dim)
    output <- torch::torch_clamp(input - tau_supp_size[[1]], min=0)
    ctx$save_for_backward(supp_size = tau_supp_size[[2]], output = output, dim = dim)
    output
  },
  backward = function(ctx, grad_output) {

    # supp_size, output = ctx$saved_variables
    saved <- ctx$saved_variables
    dim <- saved$dim
    grad_input <- grad_output$clone()
    grad_input[saved$output == 0] <- 0

    v_hat <- grad_input$sum(dim=dim) / saved$supp_size$to(dtype = saved$output$dtype)$squeeze()
    v_hat <- v_hat$unsqueeze(dim)
    grad_input <- torch::torch_where(saved$output != 0, grad_input - v_hat, grad_input)

    list(
      input = grad_input,
      dim = NULL
    )
  }
)

sparsemax <- torch::nn_module(
  "sparsemax",
  initialize = function(dim = -1) {
    self$dim <- dim
  },
  forward = function(input) {
    sparsemax_function(input, self$dim)
  }
)


.entmax_threshold_and_support <- function(input, dim) {
  sorted_input <- input$sort(dim = dim, descending = TRUE)[[1]]
  rho <- .make_ix_like(input, dim)
  mean <- sorted_input$cumsum(dim) / rho
  mean_sq <- sorted_input$pow(2)$cumsum(dim) / rho
  ss <- rho * (mean_sq - mean$pow(2))
  delta <- (1 - ss) / rho

  delta_nz <- torch::torch_clamp(delta, 0)
  tau <- mean - torch::torch_sqrt(delta_nz)

  support_size <- (tau <= sorted_input)$sum(dim)$unsqueeze(dim)
  tau_star <- tau$gather(dim, support_size)
  list(
    tau_star,
    support_size
  )
}

entmax_function <- torch::autograd_function(
  forward = function(ctx, input, dim = -1) {
    max_val <- input$max(dim=dim, keepdim=TRUE)[[1]]
    input$sub_(max_val) # same numerical stability trick as for softmax
    input <- input / 2

    tau_supp <- .entmax_threshold_and_support(input, dim=dim)
    output <- torch::torch_clamp(input - tau_supp[[1]], min=0) ^ 2
    ctx$save_for_backward(supp_size = tau_supp[[2]], output = output, dim = dim)
    output
  },
  backward = function(ctx, grad_output) {

    # supp_size, output = ctx$saved_variables
    saved <- ctx$saved_variables
    dim <- saved$dim
    Y <- saved$output
    gppr <- Y$sqrt()
    dX <- grad_output * gppr
    q <- dX$sum(dim) / gppr$sum(dim)
    q <- q$unsqueeze(dim)
    dX$sub_(q*gppr)

    list(
      input = dX,
      dim = NULL
    )
  }
)

entmax <- torch::nn_module(
  "entmax",
  initialize = function(dim = -1) {
    self$dim <- dim
  },
  forward = function(input) {
    if (self$dim == -1) {
      self$dim <- input$dim()
    }
    entmax_function(input, self$dim)
  }
)

#' Optimal threshold (tau) computation for 1.5-entmax
#'
#' @param input The input tensor to compute thresholds over.
#' @param dim The dimension along which to apply 1.5-entmax. Default is -1.
#' @param k The number of largest elements to partial-sort over. For optimal
#'   performance, should be slightly bigger than the expected number of
#'   nonzeros in the solution. If the solution is more than k-sparse,
#'   this function is recursively called with a 2*k schedule. If `NULL`, 
#'   full sorting is performed from the beginning. Default is NULL.
#'
#' @return The threshold value for each vector, with all but the `dim` 
#'   dimension intact.
#' @export
get_tau <- function(input, dim = -1, k = NULL) {
  tau_ <- ss_ <- NULL # avoid note 
  if (is.null(k) || k >= input$dim()[dim]) { # do full sort
    sorted_input <- input$sort(dim = dim, descending = TRUE)[[1]]
  } else {
    sorted_input <- input$topk(k = k, dim = dim)[[1]]
  }
  
  rho <- .make_ix_like(sorted_input, dim)
  mean <- sorted_input$cumsum(dim) / rho
  mean_sq <- sorted_input$pow(2)$cumsum(dim) / rho
  ss <- rho * (mean_sq - mean$pow(2))
  delta <- (1 - ss) / rho
  
  # NOTE this is not exactly the same as in reference algo
  # Fortunately it seems the clamped values never wrongly
  # get selected by tau <= sorted_z. Prove this!
  delta_nz <- torch::torch_clamp(delta, 0)
  tau <- mean - delta_nz$sqrt()
  
  support_size <- (tau <= sorted_input)$sum(dim)$unsqueeze(dim)
  tau_star <-  tau$gather(dim, support_size - 1)
  
  if (!is.null(k) && k < input$shape[dim]) {
    unsolved <- (support_size == k)$squeeze(dim)
    
    if (unsolved$sum() > 0) {
      input_ <- .roll_last(input, dim)[unsolved]
      c(tau_, ss_) %<-% .entmax_threshold_and_support(input_, dim = -1, k = 2 * k)
      .roll_last(tau_star, dim)[unsolved] <- tau_
      .roll_last(support_size, dim)[unsolved] <- ss_
    }
  }  
  return(tau_star)
}

relu15_function <- torch::autograd_function(
  forward = function(ctx, X, dim = 0, tau = 0) {
    logit <- torch::torch_clamp(X / 2 - tau, min = 0)
    ctx$save_for_backward(logit)
    Y <- logit$pow(2)
    return(Y)
  },
  backward = function(ctx, dY) {
    logit <- ctx$get_saved_tensors()
    return(list(logit, NULL, NULL))
  }
)

relu15 <- torch::nn_module(
  "relu15",
  initialize = function(dim = -1, tau = 0.0) {
    self$dim <- dim
    self$tau <- tau
  },
  forward = function(input) {
    return(relu15_function$apply(input, self$dim, self$tau))
  }
)