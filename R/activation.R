#' Multibranch Weighted Linear Unit (MB-WLU) nn module.
#'
#' @param input (N,*) tensor, where * means, any number of additional
#'   dimensions
#' @param alpha (float) the weight of ELU activation component.
#' @param beta (float) the weight of PRELU activation component.
#' @param gamma (float) the weight of SILU activation component.
#' @param init (float): the initial value of \eqn{a} of PRELU. Default: 0.25.
#'
#' @return an activation function computing  \eqn{\mathbf{\alpha \times ELU(input) + \beta \times PReLU(input) + \gamma \times SiLU(input)}}
#'
#' @examplesIf torch::torch_is_installed()
#' x <- torch::torch_randn(2, 2)
#' my_mbwlu <- nn_mbwlu(alpha = 0.6, beta = 0.2, gamma = 0.2)
#' mbwlu <- nn_mbwlu()
#' y <- my_mbwlu(x)
#' z <- mbwlu(x)
#' torch::torch_equal(y, z)
#' @export
nn_mbwlu <- torch::nn_module(
  "multibranch Weighted Linear Unit",
  initialize = function(alpha = 0.6, beta = 0.2, gamma = 0.2, init = 0.25) {
    self$alpha <- alpha
    self$beta <- beta
    self$gamma <- gamma
    self$init <- init
  },
  forward = function(input) {
    nnf_mbwlu(input, self$alpha, self$beta, self$gamma, self$init)
  }
)

#' Applies the Multibranch Weighted Linear Unit (MB-WLU) function, element_wise.
#' See [nn_mbwlu()] for more information.
#' @seealso [nn_mbwlu()].
#' @export
#' @rdname nn_mbwlu
nnf_mbwlu <- function(input, alpha = 0.6, beta = 0.2, gamma = 0.2,  init = 0.25) {
    alpha * torch::nnf_elu(input) +
      beta * torch::nnf_prelu(input, init) +
      gamma * torch::nnf_silu(input)

}
