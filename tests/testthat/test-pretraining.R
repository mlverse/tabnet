test_that("transpose_metrics is not adding an unnamed entry on top of the list", {

  metrics <- list(loss = 1, loss = 2, loss = 3, loss = 4)

  expect_no_error(
    tabnet:::transpose_metrics(metrics)
  )

  expect_equal(
    tabnet:::transpose_metrics(metrics),
    list(loss = c(1, 2, 3, 4))
  )

})

test_that("Unsupervised training with default config, data.frame and formula", {

  expect_no_error(
    fit <- tabnet_pretrain(x, y, epochs = 1)
  )
  expect_s3_class( fit, "tabnet_pretrain")
  expect_equal(length(fit), 3)
  expect_equal(names(fit), c("fit", "serialized_net", "blueprint"))
  expect_equal(length(fit$fit), 5)
  expect_equal(names(fit$fit), c("network", "metrics", "config", "checkpoints", "importances"))
  expect_equal(length(fit$fit$metrics), 1)

  expect_no_error(
    fit <- tabnet_pretrain(Sale_Price ~ ., data = ames, epochs = 1)
  )
  expect_s3_class( fit, "tabnet_pretrain")
  expect_equal(length(fit), 3)
  expect_equal(names(fit), c("fit", "serialized_net", "blueprint"))
  expect_equal(length(fit$fit), 5)
  expect_equal(names(fit$fit), c("network", "metrics", "config", "checkpoints", "importances"))
  expect_equal(length(fit$fit$metrics), 1)

})

test_that("Unsupervised training with pretraining_ratio", {

  expect_no_error(
    pretrain <- tabnet_pretrain(attrix, attriy, epochs = 1, pretraining_ratio=0.2)
  )

})

test_that("Unsupervised training prevent predict with an explicit message", {

  pretrain <- tabnet_pretrain(attrix, attriy, epochs = 1, pretraining_ratio=0.2)

  expect_error(
    predict(pretrain, attrix, type = "prob"),
    regexp = "tabnet_pretrain"
  )

  expect_error(
    predict(pretrain, attrix),
    regexp = "tabnet_pretrain"
  )

})

test_that("pretraining with `tabnet_model= ` parameter raise a warning", {

  expect_warning(
    fit <- tabnet_pretrain(x, y, epochs = 1, tabnet_model = ames_pretrain)
  )
  expect_s3_class( fit, "tabnet_pretrain")
  expect_equal( length(fit), length(ames_pretrain))
  expect_equal( length(fit$fit$metrics), 1)
})

test_that("errors when using an argument that do not exist", {

  expect_error(
    pretrain <- tabnet_pretrain(x, y, pretraining_ratiosas = 1-1e5),
    regexp = "unused argument"
  )

})

test_that("works with validation split", {

  expect_no_error(
    pretrain <- tabnet_pretrain(attrix, attriy, epochs = 1, valid_split = 0.2)
  )

  expect_no_error(
    pretrain <- tabnet_pretrain(attrix, attriy, epochs = 1, valid_split = 0.2, verbose = TRUE)
  )

})

test_that("works with categorical embedding dimension as list", {

  config <- tabnet_config(cat_emb_dim=c(1,1,2,2,1,1,1,2,1,1,1,2,2,2))

  expect_no_error(
    pretrain <- tabnet_pretrain(attrix, attriy, epochs = 1, valid_split = 0.2, config=config)
  )
})

test_that("explicit error message when categorical embedding dimension vector has wrong size", {

  config <- tabnet_config(cat_emb_dim=c(1,1,2,2))

  expect_error(
    pretrain <- tabnet_pretrain(attrix, attriy, epochs = 1, valid_split = 0.2, config=config),
    regexp = "number of categorical predictors"
  )
})

test_that("can train from a recipe", {

  rec <- recipe(Attrition ~ ., data = attrition) %>%
    step_normalize(all_numeric(), -all_outcomes())

  expect_no_error(
    pretrain <- tabnet_pretrain(rec, attrition, epochs = 1, verbose = TRUE)
  )

})

test_that("lr scheduler step works", {

  expect_no_error(
    fit <- tabnet_pretrain(x, y, epochs = 3, lr_scheduler = "step",
                           lr_decay = 0.1, step_size = 1)
  )

  sc_fn <- function(optimizer) {
    torch::lr_step(optimizer, step_size = 1, gamma = 0.1)
  }

  expect_no_error(
    fit <- tabnet_pretrain(x, y, epochs = 3, lr_scheduler = sc_fn,
                           lr_decay = 0.1, step_size = 1)
  )

})

test_that("lr scheduler reduce_on_plateau works", {

  expect_no_error(
    fit <- tabnet_pretrain(x, y, epochs = 3, lr_scheduler = "reduce_on_plateau",
                           lr_decay = 0.1, step_size = 1)
  )

  sc_fn <- function(optimizer) {
    torch::lr_reduce_on_plateau(optimizer, factor = 0.1, patience = 10)
  }

  expect_no_error(
    fit <- tabnet_pretrain(x, y, epochs = 3, lr_scheduler = sc_fn,
                           lr_decay = 0.1, step_size = 1)
  )

})

test_that("checkpoints works", {

  expect_no_error(
    pretrain <- tabnet_pretrain(x, y, epochs = 3, checkpoint_epochs = 1)
  )

  expect_length( pretrain$fit$checkpoints, 3  )

  # expect_equal(  pretrain$fit$checkpoints[[3]], pretrain$serialized_net )

})

test_that("print module works", {

  testthat::local_edition(3)
  testthat::skip_on_os("linux")
  testthat::skip_on_os("windows")

  expect_no_error(
    fit <- tabnet_pretrain(x, y, epochs = 1)
  )

  withr::with_options(new = c(cli.width = 50),
                      expect_snapshot_output(fit))

})

test_that("num_independent_decoder and num_shared_decoder change the network number of parameters", {

  expect_no_error(
    pretrain <- tabnet_pretrain(attrix, attriy, epochs = 1,
                                num_independent_decoder = 3, num_shared_decoder = 2)
  )
  expect_gt( torch:::get_parameter_count(pretrain$fit$network),
             torch:::get_parameter_count(attr_pretrained$fit$network)
  )
})

test_that("num_independent_decoder and num_shared_decoder do not change the network number of parameters for fit", {

  expect_no_error(
    config <- tabnet_config(epochs = 1,
                            num_independent_decoder = 3, num_shared_decoder = 2)
  )
  expect_no_error(
    attr_fit <- tabnet_fit(attrix, attriy, config = config)
  )
  expect_equal( torch:::get_parameter_count(attr_fit$fit$network),
                torch:::get_parameter_count(attr_fitted$fit$network)
  )
})

