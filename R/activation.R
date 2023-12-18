#' Multi-branch Weighted Linear Unit (MB-wLU) nn module.
#'
#' @param input (N,*) tensor, where * means, any number of additional
#'   dimensions
#' @param alpha (float) the weight of ELU activation component.
#' @param beta (float) the weight of PReLU activation component.
#' @param gamma (float) the weight of SiLU activation component.
#' @param init (float): the initial value of \eqn{a} of PReLU. Default: 0.25.
#'
#' @return an activation function computing
#' \eqn{\mathbf{MBwLU(input) = \alpha \times ELU(input) + \beta \times PReLU(input) + \gamma \times SiLU(input)}}
#'
#' @examplesIf torch::torch_is_installed()
#' x <- torch::torch_randn(2, 2)
#' my_mb_wlu <- nn_mb_wlu(alpha = 0.6, beta = 0.2, gamma = 0.2)
#' default_mb_wlu <- nn_mb_wlu()
#' y <- my_mb_wlu(x)
#' z <- default_mb_wlu(x)
#' torch::torch_equal(y, z)
#' @export
nn_mb_wlu <- torch::nn_module(
  "multibranch Weighted Linear Unit",
  initialize = function(alpha = 0.6, beta = 0.2, gamma = 0.2, init = 0.25) {
    self$alpha <- alpha
    self$beta <- beta
    self$gamma <- gamma
    self$init <- init
  },
  forward = function(input) {
    nnf_mb_wlu(input, self$alpha, self$beta, self$gamma, self$init)
  }
)

#' Applies the Multi-branch Weighted Linear Unit (MB-wLU) function, element_wise.
#' See [nn_mb_wlu()] for more information.
#' @seealso [nn_mb_wlu()].
#' @export
#' @rdname nn_mb_wlu
nnf_mb_wlu <- function(input, alpha = 0.6, beta = 0.2, gamma = 0.2,  init = 0.25) {
    alpha * torch::nnf_elu(input) +
      beta * torch::nnf_prelu(input, init) +
      gamma * torch::nnf_silu(input)

}
