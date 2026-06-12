test_that("Supervised training can continue with a additional fit, with or wo from_epoch=", {

  fit_2 <- tabnet_fit(x, y, tabnet_model = ames_fit, epochs = 1)

  expect_equal(fit_2$fit$config$epoch, 1)
  expect_length(fit_2$fit$metrics, 6)
  expect_identical(ames_fit$fit$metrics[[1]]$train, fit_2$fit$metrics[[1]]$train)
  expect_identical(ames_fit$fit$metrics[[5]]$train, fit_2$fit$metrics[[5]]$train)

  expect_no_error(
    fit_3 <- tabnet_fit(x, y, tabnet_model = ames_fit, from_epoch = 2, epoch = 1 )
  )
  expect_equal(fit_3$fit$config$epoch, 1)
  expect_length(fit_3$fit$metrics, 3)
  expect_identical(ames_fit$fit$metrics[[1]]$train, fit_2$fit$metrics[[1]]$train)
  expect_identical(ames_fit$fit$metrics[[2]]$train, fit_2$fit$metrics[[2]]$train)

})

test_that("we can change the tabnet_options between training epoch", {

  fit_2 <- tabnet_fit(x, y, ames_fit, epochs = 1, penalty = 0.003, learn_rate = 0.002)

  expect_equal(fit_2$fit$config$epoch, 1)
  expect_length(fit_2$fit$metrics, 6)
  expect_equal(fit_2$fit$config$learn_rate, 0.002)

})

test_that("epoch counter is valid for retraining from a checkpoint", {

  tmp <- tempfile("model", fileext = "rds")
  withr::local_file(saveRDS(ames_fit, tmp))

  fit1 <- readRDS(tmp)
  fit_2 <- tabnet_fit(x, y, ames_fit, epochs = 12, verbose=T)

  expect_equal(fit_2$fit$config$epoch, 12)
  expect_length(fit_2$fit$metrics, 17)
  expect_lte(mean(fit_2$fit$metrics[[17]]$train), mean(fit_2$fit$metrics[[1]]$train))

})

test_that("training loss keeps decreasing across a checkpoint boundary", {
  # Default checkpoint_epochs = 10; training to epoch 12 crosses the first checkpoint.
  # A broken device round-trip during checkpointing caused the optimizer to lose its
  # parameter references, freezing the loss at an identical floating-point value for
  # every epoch after the first checkpoint.
  fit <- tabnet_fit(Attrition ~ ., data = attrition, epochs = 12, learn_rate = 1e-2)

  expect_length(fit$fit$metrics, 12)
  train_losses <- purrr::map_dbl(fit$fit$metrics, ~mean(.x$train))

  # Loss must still be changing after the checkpoint at epoch 10: a frozen
  # optimizer repeats an exact floating-point value every epoch
  expect_false(train_losses[11] == train_losses[12])
})

test_that("training loss keeps decreasing when resuming from a disk-saved model", {
  # attr_fitted has 12 epochs with checkpoint_epochs = 10 (one checkpoint saved).
  # After disk restore, check_net_is_empty_ptr = TRUE and the code restores weights
  # from serialized_net + apply_checkpoint, then resumes training with epoch_shift = 10.
  # Before the fix, this code path could also freeze the optimizer.
  tmp <- tempfile("model", fileext = "rds")
  withr::local_file(saveRDS(attr_fitted, tmp))

  fit_from_disk <- readRDS(tmp)
  fit2 <- tabnet_fit(attrix, attriy, tabnet_model = fit_from_disk, epochs = 2)

  train_losses <- purrr::map_dbl(fit2$fit$metrics, ~mean(.x$train))
  n <- length(train_losses)

  # After disk restore, the optimizer must still be updating parameters:
  # a frozen optimizer would produce identical loss values every epoch
  expect_false(train_losses[n - 1] == train_losses[n])
  # Overall, loss after resumed training must be lower than at the very start
  expect_lte(train_losses[n], train_losses[1])
})

test_that("trying to continue training with different dataset raise error", {

  pretrain_1 <- tabnet_pretrain(x, y, epochs = 1)

  expect_error(
    pretrain_2 <- tabnet_fit(attrix, y, tabnet_model=pretrain_1, epochs = 1),
    regexp = "Model dimensions"
  )

  fit_1 <- tabnet_fit(x, y, epochs = 1)

  expect_error(
    fit_2 <- tabnet_fit(attrix, y, tabnet_model=fit_1, epochs = 1),
    regexp = "Model dimensions"
  )

  expect_error(
    fit_2 <- tabnet_fit(x, attriy, tabnet_model=fit_1, epochs = 1),
    regexp = "Model dimensions"
  )

})

test_that("Supervised training can continue unsupervised training, with or wo from_epoch=", {

  expect_no_error(
    tabnet_fit(x, y, tabnet_model = ames_pretrain, epoch = 1)
  )

  expect_no_error(
    tabnet_fit(Attrition ~ ., data = attrition, tabnet_model = attr_pretrained, epochs = 1)
  )

  expect_no_error(
    tabnet_fit(x, y, tabnet_model = ames_pretrain, from_epoch = 1, epoch = 1 )
  )

})

test_that("Supervised training can continue unsupervised training, with a Libtorch optimizer", {
  testthat::skip_if(!torch_has_optim_ignite())

  expect_no_error(
    tabnet_pretrain(x, y, epoch = 1, config = tabnet_config(
      optimizer = torch::optim_ignite_adamw)
      )
  )

  expect_no_error(
    tabnet_fit(Attrition ~ ., data = attrition, tabnet_model = attr_pretrained, epochs = 1, 
      optimizer = torch::optim_ignite_adamw
      )
  )
})


test_that("serialization of tabnet_pretrain with saveRDS just works", {

  fit <- tabnet_fit(x, y, ames_pretrain, epoch = 1, learn_rate = 1e-12)

  tmp <- tempfile("model", fileext = "rds")
  withr::local_file(saveRDS(ames_pretrain, tmp))

  pretrain2 <- readRDS(tmp)
  fit2 <- tabnet_fit(x, y, pretrain2, epoch = 1, learn_rate = 1e-12)

  expect_equal(
    predict(fit, ames),
    predict(fit2, ames),
    tolerance = 20
  )

  expect_equal(as.numeric(fit2$fit$network$.check), 1)

})
