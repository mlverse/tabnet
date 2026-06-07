#' Self-supervised learning loss
#'
#' Creates a criterion that measures the Autoassociative self-supervised learning loss between each
#' element in the input \eqn{y_pred} and target \eqn{embedded_x} on the values masked by \eqn{obfuscation_mask}. 
#' 
#' @noRd
#' @importFrom torch torch_mul torch_std torch_matmul torch_mean
nn_unsupervised_loss <- nn_module(
  "nn_unsupervised_loss",
  inherit = torch::nn_cross_entropy_loss,
  
  initialize = function(eps = 1e-9){
    super$initialize()
    self$eps = eps
  },
  
  forward = function(y_pred, embedded_x, obfuscation_mask){
    errors <- y_pred - embedded_x
    reconstruction_errors <- torch_mul(errors, obfuscation_mask) ^ 2
    batch_stds <- torch_std(embedded_x, dim = 1) ^ 2 + self$eps
    
    # compute the number of obfuscated variables to reconstruct
    nb_reconstructed_variables <- torch_sum(obfuscation_mask, dim = 2)
    
    # take the mean of the reconstructed variable errors
    features_loss <- torch_matmul(reconstruction_errors, 1 / batch_stds) / (nb_reconstructed_variables +  self$eps)
    loss <- torch_mean(features_loss, dim = 1)
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
#' @importFrom torch nn_module torch_sum torch_cat torch_minimum torch_long torch_argsort
#' @importFrom torch torch_tensor as_array
nn_aum_loss <- nn_module(
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
    if(as.logical(torch_sum(is_positive) == 0) || as.logical(torch_sum(is_negative) == 0)){
      return(torch_sum(pred_tensor*0))
    }
    
    # pred tensor may be [prediction, case_wts] when add_case_weight() is used. We keep only prediction
    if (pred_tensor$ndim > label_tensor$ndim) {
      pred_tensor <- pred_tensor$slice(dim = 2, 0, 1)$squeeze(2) 
    }
    
    # nominal case
    fn_diff <- -1L * is_positive
    fp_diff <- is_negative$to(dtype = torch_long())
    fp_denom <- torch_sum(is_negative) # or 1 for AUM based on count instead of rate
    fn_denom <- torch_sum(is_positive) # or 1 for AUM based on count instead of rate
    sorted_pred_ids <- torch_argsort(pred_tensor, dim = 1, descending = TRUE)$squeeze(-1)
    
    sorted_fp_cum <- fp_diff[sorted_pred_ids]$cumsum(dim = 1) / fp_denom
    sorted_fn_cum <- -fn_diff[sorted_pred_ids]$flip(1)$cumsum(dim = 1)$flip(1) / fn_denom
    sorted_thresh_gr <- -pred_tensor[sorted_pred_ids]
    sorted_dedup <- sorted_thresh_gr$diff(dim = 1) != 0
    # pad to replace removed last element
    padding <- sorted_dedup$slice(dim = 1, 0, 1) # torch_tensor 1 w same dtype, same shape, same device 
    sorted_fp_end <- torch_cat(c(sorted_dedup, padding))
    sorted_fn_end <- torch_cat(c(padding, sorted_dedup))
    uniq_thresh_gr <- sorted_thresh_gr[sorted_fp_end]
    uniq_fp_after <- sorted_fp_cum[sorted_fp_end]
    uniq_fn_before <- sorted_fn_cum[sorted_fn_end]
    if (pred_tensor$ndim == 1) {
      FPR <- torch_cat(c(padding$logical_not(), uniq_fp_after)) # FPR with trailing 0
      FNR <-  torch_cat(c(uniq_fn_before, padding$logical_not())) # FNR with leading 0
      self$roc_aum <- list(
        FPR = FPR,
        FNR = FNR,
        TPR = 1 - FNR,
        "min(FPR,FNR)" = torch_minimum(FNR, FPR), # full-range min(FNR, FPR)
        constant_range_low = torch_cat(c(torch_tensor(-Inf), uniq_thresh_gr)),
        constant_range_high = torch_cat(c(uniq_thresh_gr, torch_tensor(Inf)))
      ) %>% purrr::map_dfc(as_array)
    }
    min_FPR_FNR <- torch_minimum(uniq_fp_after[1:-2], uniq_fn_before[2:N])
    constant_range_gr <- uniq_thresh_gr$diff() # range splits leading to {FPR, FNR } errors (see roc_aum row)
    torch_sum(min_FPR_FNR * constant_range_gr, dim = 1)
    
  }
)


#' Apply hierarchy constraints via max-pooling over descendants (MCM)
#'
#' Given neural network outputs x and ancestor matrix R, enforces that
#' if a class is predicted positive, all its ancestors must also be positive.
#' Implements: `final_out[i] = max{x[j] : R[i,j] = 1}`
#'
#' @param x A `torch_tensor` of shape `(batch_size, tot_levels)`.
#' @param R A `torch_tensor` of shape `(1, tot_levels, tot_levels)` where 
#'   `R[1, i, j] = 1` iff class `i` is a descendant of class `j`.
#' @return A `torch_tensor` of shape `(batch_size, tot_levels)` with constrained outputs.
get_constr_output <- function(x, R) {
  c_out <- x$double()$unsqueeze(2)$expand(c(x$shape[1], R$shape[2], R$shape[2]))
  R_batch <- R$expand(c(x$shape[1], R$shape[2], R$shape[2]))
  final_out <- (R_batch * c_out)$clone()$max(dim = 3)
  final_out[[1]]
}


#' Max-Constraint Margin Loss (functional)
#'
#' Computes the hierarchy-constrained loss for multi-label classification.
#' Enforces that if a class is predicted positive, all its ancestors must 
#' also be positive, using the ancestor matrix R.
#'
#' The loss combines constrained outputs differently for positive and negative 
#' labels:
#' \itemize{
#'   \item For positive labels: uses constrained output of label-weighted predictions
#'   \item For negative labels: uses constrained raw predictions (penalizes ancestor violations)
#' }
#'
#' @param output A `torch_tensor` of raw network outputs (pre-sigmoid), 
#'   shape `(batch_size, tot_levels)`.
#' @param target Binary target labels, shape `(batch_size, tot_levels)`.
#' @param R Ancestor matrix tensor of shape `(1, tot_levels, tot_levels)` where 
#'   `R[1, i, j] = 1` iff class `i` is a descendant of class `j`.
#' @param to_eval Optional logical tensor of shape `(tot_levels,)` indicating 
#'   which classes to include in the loss computation. If `NULL`, all classes 
#'   are evaluated.
#' @param criterion Loss function to apply after constraint propagation. 
#'   Default: `nnf_binary_cross_entropy_with_logits` (expects raw logits).
#'
#' @return A scalar `torch_tensor` containing the computed loss, or a tensor 
#'   of shape `(batch_size, tot_levels)` if `reduction = "none"`.
#'
#' @seealso [nn_mc_loss()], [get_constr_output()]
#' @export
#' @importFrom torch nnf_binary_cross_entropy_with_logits
nnf_mc_loss <- function(output, target, R, to_eval = NULL, 
                        criterion = nnf_binary_cross_entropy_with_logits) {
  # Ensure double precision for numerical stability during constraint propagation
  output_d <- output$double()
  
  # 1. Constrained output from raw predictions: max-pool over descendants
  constr_output <- get_constr_output(output_d, R)  # (batch, tot_levels)
  
  # 2. Label-weighted output, then constrained (for positive label handling)
  labeled_output <- target * output_d
  train_output <- get_constr_output(labeled_output, R)
  
  # 3. Blend outputs based on ground-truth labels:
  #    - Positive labels: use constrained label-weighted output
  #    - Negative labels: use constrained raw output
  blended_output <- (1 - target) * constr_output + target * train_output
  
  # 4. Select classes to evaluate (if specified)
  if (!is.null(to_eval)) {
    blended_output <- blended_output[, to_eval, drop = FALSE]
    target <- target[, to_eval, drop = FALSE]
  }
  
  # 5. Apply the base loss function (e.g., BCE with logits)
  loss <- criterion(
    blended_output, 
    target$double()
  )
  
  return(loss)
}


#' Max-Constraint Margin Loss (module)
#'
#' Module wrapper for [nnf_mc_loss()] with configurable parameters.
#' Stores the ancestor matrix R and evaluation mask for reuse across batches.
#'
#' @param R Ancestor matrix tensor of shape `(1, tot_levels, tot_levels)`.
#' @param to_eval Optional logical tensor of shape `(tot_levels,)` indicating 
#'   which classes to include in loss computation.
#' @param criterion Loss function module or functional to apply after constraint 
#'   propagation. Default: `nn_binary_cross_entropy_with_logits()`.
#' @param reduction (string, optional): Reduction method: `'none'` | `'mean'` | `'sum'`.
#'
#' @section Shape:
#' - Input `output`: \eqn{(N, C)} where N = batch size, C = number of classes
#' - Input `target`: \eqn{(N, C)}, same shape as output, binary values
#' - Output: scalar by default. If `reduction = "none"`, then \eqn{(N, C')} 
#'   where C' is the number of evaluated classes
#'
#' @examples
#' \dontrun{
#' # Build ancestor matrix from hierarchy
#' R <- build_ancestor_matrix_from_outcomes(my_tree, processed$outcomes, device = "cuda")
#' 
#' # Create loss module
#' loss_fn <- nn_mc_loss(R = R, reduction = "mean")
#' 
#' # Forward pass
#' output <- model(x)  # (batch, tot_levels)
#' loss <- loss_fn(output, labels)
#' loss$backward()
#' }
#'
#' @seealso [nnf_mc_loss()], [build_ancestor_matrix_from_outcomes()], [get_constr_output()]
#' @export
nn_mc_loss <- nn_module(
  "nn_mc_loss",
  inherit = torch::nn_l1_loss,
  
  initialize = function(R, to_eval = NULL, 
                        criterion = torch::nnf_binary_cross_entropy_with_logits,
                        reduction = "mean") {
    super$initialize(reduction = reduction)
    
    # Store ancestor matrix (move to device if needed)
    self$R <- R
    self$to_eval <- to_eval
    # Resolve criterion based on its type
    self$criterion_fn <- .resolve_mc_criterion(criterion, reduction)
  },
  
  forward = function(output, target) {
    nnf_mc_loss(
      output = output,
      target = target,
      R = self$R,
      to_eval = self$to_eval,
      criterion = self$criterion_fn
    )
  }
)

#' Resolve criterion into a callable function(input, target, reduction)
#' @keywords internal
#' @noRd
.resolve_mc_criterion <- function(criterion, reduction) {
  # Case 1: Already an nn_module instance
  if (inherits(criterion, "nn_module")) {
    module_reduction <- criterion$reduction
    if (!is.null(module_reduction) && module_reduction != reduction) {
      warn(
        c(
          "The criterion module has reduction={.val {module_reduction}}",
          "but nn_mc_loss was called with reduction={.val {reduction}}.",
          "i" = "The module's reduction will be used."
        ),
        class = "mc_loss_reduction_mismatch"
      )
    }
    return(function(input, target) criterion(input, target))
  }
  
  # Case 2: A function (could be functional nnf_* or constructor nn_*)
  if (rlang::is_function(criterion)) {
    # Try to detect if it's a constructor by calling with just reduction
    # Constructors return nn_module, functionals need input/target
    maybe_module <- tryCatch(
      {
        result <- criterion(reduction = reduction)
        if (inherits(result, "nn_module")) result else NULL
      },
      error = function(e) NULL
    )
    
    if (!is.null(maybe_module)) {
      # It's a constructor (e.g., nn_bce_with_logits_loss)
      return(function(input, target) maybe_module(input, target))
    }
    
    # It's a functional (e.g., nnf_binary_cross_entropy_with_logits)
    return(function(input, target) {
      criterion(input, target, reduction = reduction)
    })
  }
  
  # Invalid type
  value_error(
    c(
      "`criterion` must be a function or an `nn_module`.",
      "x" = "Got: {.class {class(criterion)[1]}}"
    ),
    class = "mc_loss_invalid_criterion"
  )
}

#' Convert class_id tensor to binary one-hot tensor
#'
#' Transforms a tensor of class indices (one column per hierarchy level)
#' into a binary tensor where each column corresponds to a class.
#'
#' @param y A `torch_tensor` of shape `(batch_size, n_outcome)` containing
#'   1-based class indices.
#' @param outcomes A tibble with factor columns (as from `hardhat::mold()$outcomes`).
#' @param device Torch device.
#' @return A `torch_tensor` of shape `(batch_size, tot_levels)` with binary values.
#' @export
nnf_multilabel_one_hot <- function(y, outcomes, device = "cpu") {
  batch_size <- y$shape[1]
  n_outcome <- y$shape[2]
  
  # Number of classes per level
  n_levels <- sapply(outcomes, nlevels)
  tot_levels <- sum(n_levels)
  
  one_hot_list <- vector("list", n_outcome)
  
  for (outcome in seq_len(n_outcome)) {
    level_ids <- y[, outcome]$to(dtype = torch::torch_long())
    
    # Encode one-hot of each levels
    one_hot_list[[outcome]] <- torch::nnf_one_hot(
      level_ids, 
      num_classes = n_levels[outcome]
    )
  }
  # concatenate along the columns axis)
  torch::torch_cat(one_hot_list, dim = 2)$to(
    dtype = torch::torch_double(), 
    device = device
  )
}

