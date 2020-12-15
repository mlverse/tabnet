.onLoad <- function(...) {
  vctrs::s3_register("parsnip::multi_predict", "_tabnet_fit")
}
