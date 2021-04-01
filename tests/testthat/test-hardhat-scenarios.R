test_that("Supervised training can continue with a additional fit, with or wo from_epoch=", {

  data("ames", package = "modeldata")
  ids <- sample(nrow(ames), 256)
  x <- ames[ids,-which(names(ames) == "Sale_Price")]
  y <- ames[ids,]$Sale_Price

  fit_1 <- tabnet_fit(x, y, epochs = 5 , checkpoint_epochs = 2)
  fit_2 <- tabnet_fit(x, y, tabnet_model = fit_1, epochs = 1)

  expect_equal(fit_2$fit$config$epoch, 1)
  expect_length(fit_2$fit$metrics, 6)
  expect_identical(fit_1$fit$metrics[[1]]$train$loss, fit_2$fit$metrics[[1]]$train$loss)
  expect_identical(fit_1$fit$metrics[[5]]$train$loss, fit_2$fit$metrics[[5]]$train$loss)

  expect_error(
    fit_3 <- tabnet_fit(x, y, tabnet_model = fit_1, from_epoch = 2, epoch = 1 ),
    regexp = NA
  )
  expect_equal(fit_3$fit$config$epoch, 1)
  expect_length(fit_3$fit$metrics, 3)
  expect_identical(fit_1$fit$metrics[[1]]$train$loss, fit_2$fit$metrics[[1]]$train$loss)
  expect_identical(fit_1$fit$metrics[[2]]$train$loss, fit_2$fit$metrics[[2]]$train$loss)

})

test_that("we can change the tabnet_options between training epoch", {

  data("ames", package = "modeldata")
  ids <- sample(nrow(ames), 256)
  x <- ames[ids,-which(names(ames) == "Sale_Price")]
  y <- ames[ids,]$Sale_Price

  fit_1 <- tabnet_fit(x, y, epochs = 1)
  fit_2 <- tabnet_fit(x, y, fit_1, epochs = 1, penalty = 0.003, learn_rate = 0.002)

  expect_equal(fit_2$fit$config$epoch, 1)
  expect_length(fit_2$fit$metrics, 2)
  expect_equal(fit_2$fit$config$learn_rate, 0.002)

})

test_that("epoch counter is valid for retraining from a checkpoint", {

  data("ames", package = "modeldata")
  ids <- sample(nrow(ames), 256)
  x <- ames[ids,-which(names(ames) == "Sale_Price")]
  y <- ames[ids,]$Sale_Price

  fit_1 <- tabnet_fit(x, y, epochs = 12, verbose=T)
  tmp <- tempfile("model", fileext = "rds")
  saveRDS(fit_1, tmp)

  fit1 <- readRDS(tmp)
  fit_2 <- tabnet_fit(x, y, fit1, epochs = 12, verbose=T)

  expect_equal(fit_2$fit$config$epoch, 12)
  expect_length(fit_2$fit$metrics, 22)
  expect_lte(mean(fit_2$fit$metrics[[22]]$train$loss), mean(fit_2$fit$metrics[[1]]$train$loss))

})

test_that("trying to continue training with different dataset raise error", {

  data("ames", package = "modeldata")
  ids <- sample(nrow(ames), 256)
  x1 <- ames[ids,-which(names(ames) == "Sale_Price")]
  y1 <- ames[ids,]$Sale_Price
  data("attrition", package = "modeldata")
  ids <- sample(nrow(attrition), 256)
  x2 <- attrition[ids,-which(names(attrition) == "Attrition")]
  y2 <- attrition$Attrition

  pretrain_1 <- tabnet_pretrain(x1, y1, epochs = 1)

  expect_error(
    pretrain_2 <- tabnet_fit(x2, y1, tabnet_model=pretrain_1, epochs = 1),
    regexp = "Model dimensions"
  )

  fit_1 <- tabnet_fit(x1, y1, epochs = 1)

  expect_error(
    fit_2 <- tabnet_fit(x2, y1, tabnet_model=fit_1, epochs = 1),
    regexp = "Model dimensions"
  )

  expect_error(
    fit_2 <- tabnet_fit(x1, y2, tabnet_model=fit_1, epochs = 1),
    regexp = "Model dimensions"
  )

})

test_that("Supervised training can continue unsupervised training, with or wo from_epoch=", {

  data("attrition", package = "modeldata")

  x <- attrition[-which(names(attrition) == "Attrition")]
  y <- attrition$Attrition
  pretrain <- tabnet_pretrain(x, y, epoch = 2, checkpoint_epochs = 1)

  expect_error(
    fit <- tabnet_fit(x, y, tabnet_model = pretrain, epoch = 1),
    regexp = NA
  )

  expect_error(
    fit <- tabnet_fit(Attrition ~ ., data = attrition, tabnet_model = pretrain, epochs = 1),
    regexp = NA
  )

  expect_error(
    fit <- tabnet_fit(x, y, tabnet_model = pretrain, from_epoch = 1, epoch = 1 ),
    regexp = NA
  )

})

test_that("serialization of tabnet_pretrain with saveRDS just works", {

  data("ames", package = "modeldata")

  ids <- sample(nrow(ames), 256)
  x <- ames[ids,-which(names(ames) == "Sale_Price")]
  y <- ames[ids,]$Sale_Price

  pretrain <- tabnet_pretrain(x, y, epochs = 1)
  fit <- tabnet_fit(x, y, pretrain, epoch = 1 )
  tmp <- tempfile("model", fileext = "rds")
  saveRDS(pretrain, tmp)

  pretrain2 <- readRDS(tmp)
  fit2 <- tabnet_fit(x, y, pretrain2, epoch = 1 )

  expect_equal(
    predict(fit, ames),
    predict(fit2, ames),
    tolerance = 10
  )

  expect_equal(as.numeric(fit2$fit$network$.check), 1)

})

