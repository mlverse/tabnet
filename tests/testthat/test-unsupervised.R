test_that("Unsupervised training with default config", {

  data("ames", package = "modeldata")

  x <- ames[-which(names(ames) == "Sale_Price")]
  y <- ames$Sale_Price

  expect_error(
    fit <- tabnet_pretrain(x, y, epochs = 1),
    regexp = NA
  )

  expect_error(
    fit <- tabnet_pretrain(Sale_Price ~ ., data = ames, epochs = 1),
    regexp = NA
  )

})

test_that("Supervised training can continue unsupervised training", {

  data("ames", package = "modeldata")

  x <- ames[-which(names(ames) == "Sale_Price")]
  y <- ames$Sale_Price
  pretrain <- tabnet_pretrain(x, y, epochs = 1)

  expect_error(
    fit <- tabnet_fit(x, y, pretrain, epoch = 1),
    regexp = NA
  )

  expect_error(
    fit <- tabnet_fit(Sale_Price ~ ., data = ames, tabnet_model = pretrain, epochs = 1),
    regexp = NA
  )

})

test_that("Unsupervised training with pretraining_ratio", {

  data("attrition", package = "modeldata")

  x <- attrition[-which(names(attrition) == "Attrition")]
  y <- attrition$Attrition

  expect_error(
    pretrain <- tabnet_pretrain(x, y, epochs = 1, pretraining_ratio=0.2),
    regexp = NA
  )

})

test_that("Unsupervised training prevent predict with an explicit message", {

  data("attrition", package = "modeldata")

  x <- attrition[-which(names(attrition) == "Attrition")]
  y <- attrition$Attrition

  expect_error(
    pretrain <- tabnet_pretrain(x, y, epochs = 1, pretraining_ratio=0.2),
    regexp = "Error"
  )

  expect_error(
    predict(pretrain, x, type = "prob"),
    regexp = "Error"
  )

  expect_error(
    predict(pretrain, x),
    regexp = "Error"
  )

})

test_that("errors when using an argument that do not exist", {

  data("ames", package = "modeldata")

  x <- ames[-which(names(ames) == "Sale_Price")]
  y <- ames$Sale_Price

  expect_error(
    pretrain <- tabnet_pretrain(x, y, pretraining_ratiosas = 1-1e5),
    "unused argument"
  )

})

test_that("works with validation split", {

  data("attrition", package = "modeldata")

  x <- attrition[-which(names(attrition) == "Attrition")]
  y <- attrition$Attrition

  expect_error(
    pretrain <- tabnet_pretrain(x, y, epochs = 1, valid_split = 0.2),
    regexp = NA
  )

  expect_error(
    pretrain <- tabnet_pretrain(x, y, epochs = 1, valid_split = 0.2, verbose = TRUE),
    regexp = NA
  )

})

test_that("can train from a recipe", {

  library(recipes)
  data("attrition", package = "modeldata")

  rec <- recipe(Attrition ~ ., data = attrition) %>%
    step_normalize(all_numeric(), -all_outcomes())

  expect_error(
    pretrain <- tabnet_pretrain(rec, attrition, epochs = 1, valid_split = 0.25,
                    verbose = TRUE),
    regexp = NA
  )

})

test_that("data-frame with missing value makes training fails with explicit message", {

  data("attrition", package = "modeldata")
  ids <- sample(nrow(attrition), 256)

  x <- attrition[ids,-which(names(attrition) == "Attrition")]
  y <- attrition[ids,]$Attrition
  y_missing <- y
  y_missing[1] <- NA

  # numerical missing
  x_missing <- x
  x_missing[1,"Age"] <- NA

  expect_error(
    miss_pretrain <- tabnet_pretrain(x_missing, y, epochs = 1),
    regexp = "missing"
  )

  # categorical missing
  x_missing <- x
  x_missing[1,"BusinessTravel"] <- NA

  expect_error(
    miss_pretrain <- tabnet_pretrain(x_missing, y, epochs = 1),
    regexp = "missing"
  )

  # missing in outcome
  expect_error(
    miss_pretrain <- tabnet_pretrain(x, y_missing, epochs = 1),
    regexp = NA
  )

})


test_that("serialization with saveRDS just works", {

  data("ames", package = "modeldata")

  x <- ames[-which(names(ames) == "Sale_Price")]
  y <- ames$Sale_Price

  pretrain <- tabnet_pretrain(x, y, epochs = 1)
  fit <- tabnet_fit(x, y, pretrain, epoch = 1 )
  tmp <- tempfile("model", fileext = "rds")
  saveRDS(pretrain, tmp)

  pretrain2 <- readRDS(tmp)
  fit2 <- tabnet_fit(x, y, pretrain2, epoch = 1 )

  expect_equal(
    predict(fit, ames),
    predict(fit2, ames)
  )

  expect_equal(as.numeric(fit2$fit$network$.check), 1)

})

test_that("scheduler works", {

  data("ames", package = "modeldata")

  x <- ames[-which(names(ames) == "Sale_Price")]
  y <- ames$Sale_Price

  expect_error(
    fit <- tabnet_pretrain(x, y, epochs = 3, lr_scheduler = "step",
                      lr_decay = 0.1, step_size = 1),
    regexp = NA
  )

  sc_fn <- function(optimizer) {
    torch::lr_step(optimizer, step_size = 1, gamma = 0.1)
  }

  expect_error(
    fit <- tabnet_pretrain(x, y, epochs = 3, lr_scheduler = sc_fn,
                      lr_decay = 0.1, step_size = 1),
    regexp = NA
  )

})

test_that("checkpoints works", {

  data("ames", package = "modeldata")

  x <- ames[-which(names(ames) == "Sale_Price")]
  y <- ames$Sale_Price

  expect_error(
    fit <- tabnet_pretrain(x, y, epochs = 3, checkpoint_epochs = 1),
    regexp = NA
  )

  expect_error(
    p1 <- predict(fit, x, epoch = 1),
    regexp = NA
  )

  expect_error(
    p2 <- predict(fit, x, epoch = 2),
    regexp = NA
  )

  expect_error(
    p3 <- predict(fit, x, epoch = 3),
    regexp = NA
  )

  expect_equal(p3, predict(fit, x))

})

test_that("print module works", {

  testthat::local_edition(3)
  testthat::skip_on_os("linux")
  testthat::skip_on_os("windows")

  data("ames", package = "modeldata")

  x <- ames[-which(names(ames) == "Sale_Price")]
  y <- ames$Sale_Price

  expect_error(
    fit <- tabnet_pretrain(x, y, epochs = 1),
    regexp = NA
  )

  withr::with_options(new = c(cli.width = 50),
                      expect_snapshot_output(fit))

})

