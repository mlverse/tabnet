test_that("Autoplot with unsupervised training, w and wo valid_split", {

  data("attrition", package = "modeldata")
  ids <- sample(nrow(attrition), 256)

  x <- attrition[ids,-which(names(attrition) == "Attrition")]
  y <- attrition[ids,]$Attrition

  tabnet_pretrained <- tabnet_pretrain(x, y, epochs = 12)
  expect_error(
    p <- autoplot(tabnet_pretrained),
    regexp = NA
  )

  tabnet_pretrained <- tabnet_pretrain(x, y, epochs = 12, valid_split=0.3)
  expect_error(
    p <- autoplot(tabnet_pretrained),
    regexp = NA
  )

})

test_that("Autoplot with supervised training, w and wo valid_split", {

  data("attrition", package = "modeldata")
  ids <- sample(nrow(attrition), 256)

  x <- attrition[ids,-which(names(attrition) == "Attrition")]
  y <- attrition[ids,]$Attrition

  tabnet_fitted <- tabnet_fit(x, y, epochs = 12)
  expect_error(
    p <- autoplot(tabnet_fitted),
    regexp = NA
  )

  tabnet_fitted <- tabnet_fit(x, y, epochs = 12, valid_split=0.3)
  expect_error(
    p <- autoplot(tabnet_fitted),
    regexp = NA
  )

})

test_that("Autoplot without checkpoint", {

  data("attrition", package = "modeldata")
  ids <- sample(nrow(attrition), 256)

  x <- attrition[ids,-which(names(attrition) == "Attrition")]
  y <- attrition[ids,]$Attrition

  tabnet_pretrain <- tabnet_pretrain(x, y, epochs = 3)
  expect_error(
    p <- autoplot(tabnet_pretrain),
    regexp = NA
  )

  tabnet_pretrain <- tabnet_pretrain(x, y, epochs = 3, valid_split=0.3)
  expect_error(
    p <- autoplot(tabnet_pretrain),
    regexp = NA
  )

  tabnet_fit <- tabnet_fit(x, y, epochs = 3)
  expect_error(
    p <- autoplot(tabnet_fit),
    regexp = NA
  )

  tabnet_fit <- tabnet_fit(x, y, epochs = 3, valid_split=0.3)
  expect_error(
    p <- autoplot(tabnet_fit),
    regexp = NA
  )

})

test_that("Autoplot of pretrain then fit scenario", {

  data("attrition", package = "modeldata")
  ids <- sample(nrow(attrition), 256)

  x <- attrition[ids,-which(names(attrition) == "Attrition")]
  y <- attrition[ids,]$Attrition

  tabnet_pretrain <- tabnet_pretrain(x, y, epochs = 12, valid_split=.2)
  tabnet_fit <- tabnet_fit(x, y, tabnet_model=tabnet_pretrain, epochs = 12)

  expect_error(
    p <- autoplot(tabnet_fit),
    regexp = NA
  )

})
