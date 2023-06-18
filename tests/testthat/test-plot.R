
test_that("Autoplot with unsupervised training, w and wo valid_split", {

  expect_no_error(
    print(autoplot(attr_pretrained))
  )

  expect_no_error(
    print(autoplot(attr_pretrained_vsplit))
  )

})

test_that("Autoplot with supervised training, w and wo valid_split", {

  expect_no_error(
    print(autoplot(attr_fitted))
  )

  expect_no_error(
    print(autoplot(attr_fitted_vsplit))
  )

})

test_that("Autoplot a model without checkpoint", {

  tabnet_pretrain <- tabnet_pretrain(attrix, attriy, epochs = 3)
  expect_no_error(
    print(autoplot(tabnet_pretrain))
  )

  tabnet_pretrain <- tabnet_pretrain(attrix, attriy, epochs = 3, valid_split=0.3)
  expect_no_error(
    print(autoplot(tabnet_pretrain))
  )

  tabnet_fit <- tabnet_fit(attrix, attriy, epochs = 3)
  expect_no_error(
    print(autoplot(tabnet_fit))
  )

  tabnet_fit <- tabnet_fit(attrix, attriy, epochs = 3, valid_split=0.3)
  expect_no_error(
    print(autoplot(tabnet_fit))
  )

})

test_that("Autoplot of pretrain then fit scenario", {

  tabnet_fit <- tabnet_fit(attrix, attriy, tabnet_model=attr_pretrained_vsplit, epochs = 12)

  expect_no_error(
    print(autoplot(tabnet_fit))
  )

})

test_that("Autoplot of tabnet_explain works for pretrain and fitted model", {

  explain_pretrain <- tabnet_explain(attr_pretrained_vsplit, attrix)
  explain_fit <- tabnet_explain(attr_fitted_vsplit, attrix)

  expect_no_error(
    print(autoplot(explain_pretrain))
  )

  expect_no_error(
    print(autoplot(explain_pretrain, type = "steps"))
  )

  expect_no_error(
    print(autoplot(explain_pretrain, type = "steps", quantile = 0.99)),

  )

  expect_no_error(
    print(autoplot(explain_fit))
  )

  expect_no_error(
    print(autoplot(explain_fit, type = "steps"))
  )

  expect_no_error(
    print(autoplot(explain_fit, type = "steps", quantile = 0.99))
  )

})

test_that("Autoplot of multi-outcome regression explainer", {

  x <- small_ames[,-which(names(ames) %in% c("Sale_Price", "Pool_Area"))]
  y <- small_ames[, c("Sale_Price", "Pool_Area")]
  ames_fit <- tabnet_fit(x, y, epochs = 5, verbose=TRUE)
  ames_explain <- tabnet_explain(ames_fit, ames)

  expect_no_error(
    print(autoplot(ames_explain))
  )

  expect_no_error(
    print(autoplot(ames_explain, type = "steps", quantile = 0.99))
  )

})
