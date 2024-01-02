.onLoad <- function(...) {
  vctrs::s3_register("parsnip::multi_predict", "_tabnet_fit")
  vctrs::s3_register("vip::vi_model", "tabnet_fit")
  vctrs::s3_register("vip::vi_model", "tabnet_pretrain")
  vctrs::s3_register("ggplot2::autoplot", "tabnet_fit")
  vctrs::s3_register("ggplot2::autoplot", "tabnet_pretrain")
  vctrs::s3_register("ggplot2::autoplot", "tabnet_explain")
  vctrs::s3_register("tune::min_grid", "tabnet")
}


globalVariables(c("batch_size",
                  "dataset",
                  "epoch",
                  "has_checkpoint",
                  "loss",
                  "M_explain_emb_dim",
                  "M_loss",
                  "mask_agg",
                  "masks_emb_dim",
                  "mean_loss",
                  "out",
                  "row_number",
                  "rowname",
                  "step",
                  "value",
                  "variable",
                  ".."))
