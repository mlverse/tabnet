
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
#' @param k The n largest elements to partial-sort input over. For optimal
#' performance, should be slightly bigger than the expected number of
#' non-zeros in the solution. If the solution is more than k-sparse,
#' this function is recursively called with a 2*k schedule. If `NULL`, full
#' sorting is performed from the beginning.
#'
#' @return A list containing:
#' \itemize{
#' \item{\code{tau}}{The threshold value for each vector, with all but the \code{dim} dimension intact.}
#' \item{\code{support_size}}{The number of non-zeros in each vector.}
#' }
#' @noRd
.sparsemax_threshold_and_support <- function(input, dim = -1L, k = NULL) {
  tau_ <- ss_ <- NULL # avoid NOTE
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
  
  if (!is.null(k) && k < input$size(dim)) {
    unsolved <- (support_size == k)$squeeze(dim)
    
    if (as.numeric(unsolved$sum()) > 0) {
      input_ <- .roll_last(input, dim)[unsolved]
      c(tau_, ss_) %<-% .sparsemax_threshold_and_support(input_, dim = -1L, k = 2 * k)
      tau_rl <- .roll_last(tau, dim)
      tau_rl[unsolved] <- tau_
      tau <- .roll_last(tau_rl, dim)
      support_size_rl <- .roll_last(support_size, dim)
      support_size_rl[unsolved] <- ss_
      support_size <- .roll_last(support_size_rl, dim)
      
    }
  }
  
  list(tau, support_size)
}


sparsemax_function <- torch::autograd_function(
  forward = function(ctx, input, dim = -1L, k = NULL) {
    max_val <- input$max(dim = dim, keepdim = TRUE)[[1]]
    input <- input - max_val # same numerical stability trick as for softmax
    c(tau, supp_size) %<-% .sparsemax_threshold_and_support(input, dim = dim, k = k)
    output <- torch::torch_clamp(input - tau, min = 0)
    ctx$save_for_backward(supp_size = supp_size, output = output, dim = dim, k = k)
    output
  },

  backward = function(ctx, grad_output) {
    c(supp_size, output, dim, k) %<-% ctx$saved_variables
    grad_input <- grad_output$clone()
    grad_input[output == 0] <- 0

    v_hat <- grad_input$sum(dim = dim) / supp_size$to(dtype = output$dtype)$squeeze(dim)
    v_hat <- v_hat$unsqueeze(dim)
    grad_input <- torch::torch_where(output != 0, grad_input - v_hat, grad_input)

    list(
      input = grad_input,
      dim = dim,
      k = k
    )
  }
)

#' Sparsemax 
#' 
#' Normalizing sparse transform (a la softmax).
#'
#' Solves the projection:
#' 
#' \eqn{\min_P ||input - P||_2 \text{ s.t. } P \geq0, \sum(P) ==1}
#'
#'
#' @param dim The dimension along which to apply sparsemax.
#' @param k The number of largest elements to partial-sort input over. For optimal
#' performance, `k` should be slightly bigger than the expected number of
#' non-zeros in the solution. If the solution is more than k-sparse,
#' this function is recursively called with a 2*k schedule. If `NULL`, full
#' sorting is performed from the beginning.
#'
#' @return The projection result, such that \eqn{\sum_{dim} P = 1 \forall dim} elementwise.
#'
#' @examplesIf torch::torch_is_installed()
#' \dontrun{
#' input <- torch::torch_randn(10, 5, requires_grad = TRUE)
#' # create a top3 alpha=1.5 sparsemax on last input dimension
#' nn_sparsemax <- sparsemax15(dim=1, k=3)
#' result <- nn_sparsemax(input)
#' print(result)
#' }
#' @export
sparsemax <- torch::nn_module(
  "sparsemax",
  initialize = function(dim = -1L) {
    self$dim <- dim
  },
  forward = function(input) {
    sparsemax_function(input, self$dim)
  }
)

#' Alpha-Sparsemax with alpha equal 1.5 
#' @rdname sparsemax
#' @export
sparsemax15 <- torch::nn_module(
  "sparsemax_15",
  initialize = function(dim = -1L, k=NULL) {
    self$dim <- dim
    self$k <- k
  },
  forward = function(input) {
    sparsemax_function(input, self$dim, self$k)
  }
)

#' 1.5-entmax compute with optimal threshold and support size.
#'
#' @param input The input tensor to compute thresholds over.
#' @param dim The dimension along which to apply 1.5-entmax.
#' @param k The n largest elements to partial-sort input over. For optimal
#'   performance, `k` should be slightly bigger than the expected number of
#'   non-zeros in the solution. If the solution is more than k-sparse,
#'   this function is recursively called with a 2*k schedule. If `NULL`, full
#'   sorting is performed from the beginning.
#'
#' @return A list containing:
#'   \itemize{
#'     \item{\code{tau_star}}{The threshold value for each vector, with all but the \code{dim} dimension intact.}
#'     \item{\code{support_size}}{The number of non-zeros in each vector.}
#'   }
#' @noRd
.entmax_threshold_and_support <- function(input, dim = -1L, k = NULL) {
  tau_ <- ss_ <- NULL # avoid NOTE
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
  
  if (!is.null(k) && k < input$size(dim)) {
    unsolved <- (support_size == k)$squeeze(dim)
    
    if (as.numeric(unsolved$sum()) > 0) {
      input_ <- .roll_last(input, dim)[unsolved]
      c(tau_, ss_ ) %<-% .entmax_threshold_and_support(input_, dim = -1L, k = 2 * k)
      tau_star_rl <- .roll_last(tau_star, dim)
      tau_star_rl[unsolved] <- tau_
      tau_star <- .roll_last(tau_star_rl, dim)
      support_size_rl <- .roll_last(support_size, dim)
      support_size_rl[unsolved] <- ss_
      support_size <- .roll_last(support_size_rl, dim)
    }
  }
  
  list( tau_star, support_size)
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
    output <- saved$output
    gppr <- output$sqrt()
    dX <- grad_output * gppr
    q <- dX$sum(dim) / gppr$sum(dim)
    q <- q$unsqueeze(dim)
    dX$sub_(q*gppr)

    list(
      input = dX,
      dim = dim
    )
  }
)

#' @export
#' @rdname entmax15
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

entmax_15_function <- torch::autograd_function(
  forward = function(ctx, input, dim = 1L, k = NULL) {
    max_val <- input$max(dim = dim, keepdim = TRUE)[[1]]
    input <- (input - max_val)/2 # same numerical stability trick as for softmax

    tau_star <- .entmax_threshold_and_support(input, dim = dim, k = k)[[1]]
    output <- torch::torch_clamp(input - tau_star, min = 0) ^ 2
    ctx$save_for_backward(output = output, dim = dim, k = k)
    output
  },

  backward = function(ctx, grad_output) {
    # supp_size, output = ctx$saved_variables
    s <- ctx$saved_variables
    gppr <- s$output$sqrt()
    dX <- grad_output * gppr
    q <- dX$sum(s$dim) / gppr$sum(s$dim)
    q <- q$unsqueeze(s$dim)
    dX$sub_(q*gppr)

    list(
      input = dX,
      dim = dim,
      k = s$k
    )
  }
)

#' Alpha-entmax 
#' 
#' With alpha = 1.5 and normalizing sparse transform (a la softmax).
#'
#' Solves the optimization problem:
#' \eqn{\max_p <input, P> - H_{1.5}(P) \text{ s.t. } P \geq 0, \sum(P) == 1}
#' where \eqn{H_{1.5}(P)} is the Tsallis alpha-entropy with \eqn{\alpha=1.5}.
#'
#' @param dim The dimension along which to apply 1.5-entmax.
#' @param k The number of largest elements to partial-sort input over. For optimal
#' performance, should be slightly bigger than the expected number of
#' non-zeros in the solution. If the solution is more than k-sparse,
#' this function is recursively called with a 2*k schedule. If `NULL`, full
#' sorting is performed from the beginning.
#'
#' @return The projection result P  of the same shape as input, such that
#'   \eqn{\sum_{dim} P = 1 \forall dim} elementwise.
#'
#' @examplesIf torch::torch_is_installed()
#' \dontrun{
#' input <- torch::torch_randn(10,5, requires_grad = TRUE)
#' # create a top3 alpha=1.5 entmax on last input dimension
#' nn_entmax <- entmax15(dim=-1L, k = 3)
#' result <- nn_entmax(input)
#' }
#' @export
entmax15 <- torch::nn_module(
  "entmax_15",
  initialize = function(dim = -1L, k = NULL) {
    self$dim <- dim
    self$k <- k
  },
  forward = function(input) {
    if (self$dim == -1L) {
      self$dim <- input$dim()
    }
    entmax_15_function(input, self$dim, self$k)
  }
)

#' Optimal threshold (tau) computation for 1.5-entmax
#'
#' @param input The input tensor to compute thresholds over.
#' @param dim The dimension along which to apply 1.5-entmax. Default is -1.
#' @param k The number of largest elements to partial-sort over. For optimal
#'   performance, should be slightly bigger than the expected number of
#'   non-zeros in the solution. If the solution is more than k-sparse,
#'   this function is recursively called with a 2*k schedule. If `NULL`, 
#'   full sorting is performed from the beginning. Default is NULL.
#'
#' @return The threshold value for each vector, with all but the `dim` 
#'   dimension intact.
get_tau <- function(input, dim = -1L, k = NULL) {
  tau_ <- ss_ <- NULL # avoid note 
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
  delta_nz <- torch::torch_clamp(delta, 0)
  tau <- mean - delta_nz$sqrt()
  
  support_size <- (tau <= sorted_input)$sum(dim)$unsqueeze(dim)
  tau_star <-  tau$gather(dim, support_size)
  
  if (!is.null(k) && k < input$size(dim)) {
    unsolved <- (support_size == k)$squeeze(dim)
    
    if (as.numeric(unsolved$sum()) > 0) {
      input_ <- .roll_last(input, dim)[unsolved]
      c(tau_, ss_) %<-% .entmax_threshold_and_support(input_, dim = -1L, k = 2 * k)
      tau_star[unsolved] <- tau_
      support_size[unsolved] <- ss_
    }
  }  
  return(tau_star)
}

relu15_function <- torch::autograd_function(
  forward = function(ctx, input, dim = 0L, tau = 0) {
    logit <- torch::torch_clamp(input / 2 - tau, min = 0)
    ctx$save_for_backward(logit)
    output <- logit$pow(2)
    output
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