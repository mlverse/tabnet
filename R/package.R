.onLoad <- function(...) {
  vctrs::s3_register("parsnip::multi_predict", "_tabnet_fit")
  vctrs::s3_register("vip::vi_model", "tabnet_fit")
  vctrs::s3_register("vip::vi_model", "tabnet_pretrain")
  vctrs::s3_register("ggplot2::autoplot", "tabnet_fit")
  vctrs::s3_register("ggplot2::autoplot", "tabnet_pretrain")
  vctrs::s3_register("ggplot2::autoplot", "tabnet_explain")
  vctrs::s3_register("torch::nn_prune_head", "tabnet_fit")
  vctrs::s3_register("torch::nn_prune_head", "tabnet_pretrain")
  vctrs::s3_register("tune::min_grid", "tabnet")
}


globalVariables(c("batch_size",
                  "checkpoint",
                  "dataset",
                  "epoch",
                  "has_checkpoint",
                  "loss",
                  "mask_agg",
                  "mean_loss",
                  "row_number",
                  "rowname",
                  "step",
                  "value",
                  "variable",
                  ".."))
