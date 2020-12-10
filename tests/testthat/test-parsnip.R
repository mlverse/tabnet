test_that("multiplication works", {
  library(parsnip)
  data("ames", package = "modeldata")

  expect_error(
    model <- tabnet() %>%
      set_mode("regression") %>%
      set_engine("torch"),
    regexp = NA
  )

  expect_error(
    model %>%
      fit(Sale_Price ~ ., data = ames),
    regexp = NA
  )

})
