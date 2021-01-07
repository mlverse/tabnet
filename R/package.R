.onLoad <- function(...) {
  vctrs::s3_register("parsnip::multi_predict", "_tabnet_fit")
  vctrs::s3_register("vip::vi_model", "tabnet_fit")
}
