
# credits to Yandex https://github.com/Qwicen/node/blob/master/lib/nn_utils.py
.make_ix_like <- function(input, dim = 1) {
  d <- input$size(dim)
  rho <- torch::torch_arange(start = 1, end = d, device = input$device,
                             dtype = input$dtype)
  view <- rep(1, input$dim())
  view[1] <- -1
  rho$view(view)$transpose(1, dim)
}

.threshold_and_support <- function(input, dim) {
  input_srt <- torch::torch_sort(input, descending=TRUE, dim=dim)[[1]]
  input_cumsum <- input_srt$cumsum(dim) - 1
  rhos <- .make_ix_like(input, dim)
  support <- rhos * input_srt > input_cumsum

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
  input_srt <- torch::torch_sort(input, descending=TRUE, dim=dim)[[1]]
  rho <- .make_ix_like(input, dim)
  mean <- input_srt$cumsum(dim) / rho
  mean_sq <- (input_srt ^ 2)$cumsum(dim) / rho
  ss <- rho * (mean_sq -mean ^ 2)
  delta <- (1 - ss) / rho

  delta_nz <- torch::torch_clamp(delta, 0)
  tau <- mean - torch::torch_sqrt(delta_nz)

  support_size <- (tau <= input_srt)$sum(dim)$unsqueeze(dim)
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
