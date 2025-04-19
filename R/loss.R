#' Self-supervised learning loss
#'
#' Creates a criterion that measures the Autoassociative self-supervised learning loss between each
#' element in the input \eqn{y_pred} and target \eqn{embedded_x} on the values masked by \eqn{obfuscation_mask}. 
#' 
#' @noRd
nn_unsupervised_loss <- torch::nn_module(
  "nn_unsupervised_loss",
  inherit = torch::nn_cross_entropy_loss,
  
  initialize = function(eps = 1e-9){
    super$initialize()
    self$eps = eps
  },
  
  forward = function(y_pred, embedded_x, obfuscation_mask){
    errors <- y_pred - embedded_x
    reconstruction_errors <- torch::torch_mul(errors, obfuscation_mask) ^ 2
    batch_stds <- torch::torch_std(embedded_x, dim = 1) ^ 2 + self$eps
    
    # compute the number of obfuscated variables to reconstruct
    nb_reconstructed_variables <- torch::torch_sum(obfuscation_mask, dim = 2)
    
    # take the mean of the reconstructed variable errors
    features_loss <- torch::torch_matmul(reconstruction_errors, 1 / batch_stds) / (nb_reconstructed_variables +  self$eps)
    loss <- torch::torch_mean(features_loss, dim = 1)
    loss
  }
)


#' AUM loss
#'
#' Creates a criterion that measures the Area under the \eqn{Min(FPR, FNR)} (AUM) between each
#' element in the input \eqn{pred_tensor} and target \eqn{label_tensor}. 
#' 
#' This is used for measuring the error of a binary reconstruction within highly unbalanced dataset, 
#' where the goal is optimizing the ROC curve. Note that the targets \eqn{label_tensor} should be factor
#' level of the binary outcome, i.e. with values `1L` and `2L`.
#'
#' @examplesIf torch::torch_is_installed()
#' loss <- nn_aum_loss()
#' input <- torch::torch_randn(4, 6, requires_grad = TRUE)
#' target <- input > 1.5
#' output <- loss(input, target)
#' output$backward()
#' @export
nn_aum_loss <- torch::nn_module(
  "nn_aum_loss",
  inherit = torch::nn_mse_loss,
  initialize = function(){
    super$initialize()
    self$roc_aum <- tibble::tibble()
  },
  forward = function(pred_tensor, label_tensor){
    # thanks to https://tdhock.github.io/blog/2024/auto-grad-overhead/
    is_positive <- label_tensor == label_tensor$max()
    is_negative <- is_positive$bitwise_not()
    # manage case when prediction error is null (prevent division by 0)
    if(as.logical(torch::torch_sum(is_positive) == 0) || as.logical(torch::torch_sum(is_negative) == 0)){
      return(torch::torch_sum(pred_tensor*0))
    }
    
    # pred tensor may be [prediction, case_wts] when add_case_weight() is used. We keep only prediction
    if (pred_tensor$ndim > label_tensor$ndim) {
      pred_tensor <- pred_tensor$slice(dim = 2, 0, 1)$squeeze(2) 
    }
    
    # nominal case
    fn_diff <- -1L * is_positive
    fp_diff <- is_negative$to(dtype = torch::torch_long())
    fp_denom <- torch::torch_sum(is_negative) # or 1 for AUM based on count instead of rate
    fn_denom <- torch::torch_sum(is_positive) # or 1 for AUM based on count instead of rate
    sorted_pred_ids <- torch::torch_argsort(pred_tensor, dim = 1, descending = TRUE)$squeeze(-1)
    
    sorted_fp_cum <- fp_diff[sorted_pred_ids]$cumsum(dim = 1) / fp_denom
    sorted_fn_cum <- -fn_diff[sorted_pred_ids]$flip(1)$cumsum(dim = 1)$flip(1) / fn_denom
    sorted_thresh_gr <- -pred_tensor[sorted_pred_ids]
    sorted_dedup <- sorted_thresh_gr$diff(dim = 1) != 0
    # pad to replace removed last element
    padding <- sorted_dedup$slice(dim = 1, 0, 1) # torch_tensor 1 w same dtype, same shape, same device 
    sorted_fp_end <- torch::torch_cat(c(sorted_dedup, padding))
    sorted_fn_end <- torch::torch_cat(c(padding, sorted_dedup))
    uniq_thresh_gr <- sorted_thresh_gr[sorted_fp_end]
    uniq_fp_after <- sorted_fp_cum[sorted_fp_end]
    uniq_fn_before <- sorted_fn_cum[sorted_fn_end]
    if (pred_tensor$ndim == 1) {
      FPR <- torch::torch_cat(c(padding$logical_not(), uniq_fp_after)) # FPR with trailing 0
      FNR <-  torch::torch_cat(c(uniq_fn_before, padding$logical_not())) # FNR with leading 0
      self$roc_aum <- list(
        FPR = FPR,
        FNR = FNR,
        TPR = 1 - FNR,
        "min(FPR,FNR)" = torch::torch_minimum(FNR, FPR), # full-range min(FNR, FPR)
        constant_range_low = torch::torch_cat(c(torch::torch_tensor(-Inf), uniq_thresh_gr)),
        constant_range_high = torch::torch_cat(c(uniq_thresh_gr, torch::torch_tensor(Inf)))
      ) %>% purrr::map_dfc(torch::as_array)
    }
    min_FPR_FNR <- torch::torch_minimum(uniq_fp_after[1:-2], uniq_fn_before[2:N])
    constant_range_gr <- uniq_thresh_gr$diff() # range splits leading to {FPR, FNR } errors (see roc_aum row)
    torch::torch_sum(min_FPR_FNR * constant_range_gr, dim = 1)
    
  }
)
