test_that("early stopping message get translated in french", {
  # skip on linux on ci due to missing language in image
  testthat::skip_if((testthat:::on_ci() && testthat:::system_os() == "linux"))
  testthat::skip_on_cran()
  withr::with_language(lang = "fr",
                       expect_error(
                         tabnet_fit(attrix, attriy, epochs = 200, verbose=TRUE,
                                    early_stopping_monitor="cross_validation_loss",
                                    early_stopping_tolerance=1e-7, early_stopping_patience=3, learn_rate = 0.2),
                         regexp = "n'est pas une m"
                       )
  )
})

test_that("scheduler message translated in french", {
  # skip on linux on ci due to missing language in image
  testthat::skip_if((testthat:::on_ci() && testthat:::system_os() == "linux"))
  testthat::skip_on_cran()
  withr::with_language(lang = "fr",
                       expect_error(
                         fit <- tabnet_pretrain(x, y, epochs = 3, lr_scheduler = "multiplicative",
                                           lr_decay = 0.1, step_size = 1),
                       regexp = "Seule les planifications \"step\" et"
                       )
  )
})
