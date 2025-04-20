
# credits to Yandex https://github.com/Qwicen/node/blob/master/lib/nn_utils.py
.make_ix_like <- function(input, dim = 1L) {
  d <- input$size(dim)
  rho <- torch::torch_arange(start = 1, end = d, device = input$device,
                             dtype = input$dtype)
  view <- rep(1, input$dim())
  view[1] <- -1
  rho$view(view)$transpose(1, dim)
}

.roll_last <- function(input, dim = 1L) {
  if (dim == -1L) {
    return(input)
  } else if (dim < 0) {
    dim <- input$dim() - dim
  }
  perm <- c(which(1:input$dim() != dim), dim)
  return(input$permute(perm))
}


#' Sparsemax compute with optimal threshold and support size.
#'
#' @param input The input tensor to compute thresholds over.
#' @param dim The dimension along which to apply sparsemax.
#' @param k The number of largest elements to partial-sort over. For optimal
#' performance, should be slightly bigger than the expected number of
#' nonzeros in the solution. If the solution is more than k-sparse,
#' this function is recursively called with a2*k schedule. If `NULL`, full
#' sorting is performed from the beginning.
#'
#' @return A list containing:
#' \itemize{
#' \item{\code{tau}}{The threshold value for each vector, with all but the \code{dim} dimension intact.}
#' \item{\code{support_size}}{The number of nonzeros in each vector.}
#' }
#'
#' @examples
#' # example usage
#' input <- torch::torch.randn(10,5)
#' dim <-1
#' k <-3
#' result <- .sparsemax_threshold_and_support(input, dim, k)
#' print(result)
#' @noRd
.sparsemax_threshold_and_support <- function(ctx, input, dim = -1L, k = NULL) {
  if (is.null(k) || k >= input$size(dim)) { # do full sort
    topk <- input$sort(dim = dim, descending = TRUE)[[1]]
  } else {
    topk <- input$topk(k = k, dim = dim)[[1]]
  }
  
  topk_cumsum <- topk$cumsum(dim) - 1
  rhos <- .make_ix_like(topk, dim)
  support <- rhos * topk > topk_cumsum
  
  support_size <- support$sum(dim)$unsqueeze(dim)
  tau <- topk_cumsum$gather(dim, support_size)
  tau <- tau / support_size$to(dtype = input$dtype)
  
  if (!is.null(k) & k < dim(input)[dim]) {
    unsolved <- (support_size == k)$squeeze(dim)
    
    if (unsolved$sum() > 0) {
      in_ <- .roll_last(input, dim)[unsolved]
      c(tau_, ss_) %<-% .sparsemax_threshold_and_support(in_, dim = -1L, k = 2 * k)
      .roll_last(tau, dim)[unsolved] <- tau_
      .roll_last(support_size, dim)[unsolved] <- ss_
    }
  }
  
  ctx$save_for_backward(supp_size = support_size, dim = dim)
  list(
    tau,
    support_size
  )
}


sparsemax_function <- torch::autograd_function(
  forward = function(ctx, input, dim = -1L, k = NULL) {
    max_val <- input$max(dim = dim, keepdim = TRUE)[[1]]
    input$sub_(max_val) # same numerical stability trick as for softmax
    c(tau, supp_size) %<-% .sparsemax_threshold_and_support(input, dim = dim, k = k)
    output <- torch::torch_clamp(input - tau, min = 0)
    ctx$save_for_backward(supp_size = supp_size, output = output, dim = dim)
    output
  },

  backward = function(ctx, grad_output) {
    c(supp_size, output, dim) %<-% ctx$saved_variables
    grad_input <- grad_output$clone()
    grad_input[output == 0] <- 0

    v_hat <- grad_input$sum(dim=dim) / supp_size$to(dtype = output$dtype)$squeeze(dim)
    v_hat <- v_hat$unsqueeze(dim)
    grad_input <- torch::torch_where(output != 0, grad_input - v_hat, grad_input)

    list(
      input = grad_input,
      dim = NULL,
      k = NULL
    )
  }
)

sparsemax <- torch::nn_module(
  "sparsemax",
  initialize = function(dim = -1L) {
    self$dim <- dim
  },
  forward = function(input) {
    sparsemax_function(input, self$dim)
  }
)

#' 1.5-entmax compute with optimal threshold and support size.
#'
#' @param input The input tensor to compute thresholds over.
#' @param dim The dimension along which to apply 1.5-entmax.
#' @param k The number of largest elements to partial-sort over. For optimal
#'   performance, should be slightly bigger than the expected number of
#'   nonzeros in the solution. If the solution is more than k-sparse,
#'   this function is recursively called with a 2*k schedule. If `NULL`, full
#'   sorting is performed from the beginning.
#'
#' @return A list containing:
#'   \itemize{
#'     \item{\code{tau_star}}{The threshold value for each vector, with all but the \code{dim} dimension intact.}
#'     \item{\code{support_size}}{The number of nonzeros in each vector.}
#'   }
#'
#' @examples
#' # example usage
#' input <- torch::torch.randn(10, 5)
#' dim <- 1
#' k <- 3
#' result <- .entmax_threshold_and_support(input, dim, k)
#' print(result)
#' @noRd
.entmax_threshold_and_support <- function(input, dim, k = NULL) {
  if (is.null(k) || k >= input$size(dim)) { # do full sort
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
  delta_nz <- torch::torch_clamp(delta,0)
  tau <- mean - torch::torch_sqrt(delta_nz)
  
  support_size <- (tau <= sorted_input)$sum(dim)$unsqueeze(dim)
  tau_star <- tau$gather(dim, support_size)
  
  if (!is.null(k) && k < dim(input)[dim]) {
    unsolved <- (support_size == k)$squeeze(dim)
    
    if (unsolved$sum() > 0) {
      X_ <- .roll_last(input, dim)[unsolved]
      c(tau_, ss_ ) %<-% .entmax_threshold_and_support(X_, dim = -1L, k = 2 * k)
      tau_star[unsolved] <- tau_
      support_size[unsolved] <- ss_
    }
  }
  
  list(
    tau_star,
    support_size
  )
}


entmax_function <- torch::autograd_function(
  forward = function(ctx, input, dim = -1L) {
    max_val <- input$max(dim=dim, keepdim=TRUE)[[1]]
    input$sub_(max_val) # same numerical stability trick as for softmax
    input <- input / 2

    tau_supp <- .entmax_threshold_and_support(input, dim = dim)
    output <- torch::torch_clamp(input - tau_supp[[1]], min = 0) ^ 2
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
get_tau <- function(input, dim = -1L, k = NULL) {
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
  tau_star <-  tau$gather(dim, support_size)
  
  if (!is.null(k) && k < input$size(dim)) {
    unsolved <- (support_size == k)$squeeze(dim)
    
    if (unsolved$sum() > 0) {
      input_ <- .roll_last(input, dim)[unsolved]
      c(tau_, ss_) %<-% .entmax_threshold_and_support(input_, dim = -1L, k = 2 * k)
      tau_star[unsolved] <- tau_
      support_size[unsolved] <- ss_
    }
  }  
  return(tau_star)
}

relu15_function <- torch::autograd_function(
  forward = function(ctx, X, dim = 0L, tau = 0) {
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
  initialize = function(dim = -1L, tau = 0.0) {
    self$dim <- dim
    self$tau <- tau
  },
  forward = function(input) {
    return(relu15_function$apply(input, self$dim, self$tau))
  }
)