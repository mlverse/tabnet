#' Applies the Multibranch Weighted Linear Unit (MB-WLU) function, element-wise.
#'
#' @param input (N,*) tensor, where * means, any number of additional
#'   dimensions
#' @param alpha the weight of ELU activation component.
#' @param beta the weight of PRELU activation component.
#' @param gamma the weight of SILU activation component.
#'
#' @examplesIf torch::torch_is_installed()
#' x <- torch::torch_randn(2, 2)
#' y <- nn_mnwlu(x, alpha = 0.6, beta = 0.2, gamma = 0.2)
#' z <- nn_mnwlu(x)
#' torch::torch_equal(y, z)
#' @export
nn_mbwlu <- torch::nn_module(
  "multibranch Weighted Linear Unit",
  initialize = function(alpha = 0.6, beta = 0.2, gamma = 0.2) {
    self$alpha <- alpha
    self$beta <- beta
    self$gamma <- gamma
  },
  forward = function(input) {
    self$alpha * torch::nn_elu(input, inplace=FALSE) +
      self$beta * torch::nn_prelu(input, inplace=FALSE) +
      self$gamma * torch::nn_silu(input)
  }
)
