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
    fit <- model %>%
      fit(Sale_Price ~ ., data = ames),
    regexp = NA
  )

})

test_that("multi_predict works as expected", {

  library(parsnip)

  model <- tabnet() %>%
    set_mode("regression") %>%
    set_engine("torch", checkpoint_epochs = 1)

  data("ames", package = "modeldata")

  expect_error(
    fit <- model %>%
      fit(Sale_Price ~ ., data = ames),
    regexp = NA
  )

  preds <- multi_predict(fit, ames, epochs = c(1,2,3,4,5))

  expect_equal(nrow(preds), nrow(ames))
  expect_equal(nrow(preds$.pred[[1]]), 5)
})
