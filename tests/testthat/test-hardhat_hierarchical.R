
test_that("Training hierarchical multi-class for data.frame", {

  expect_no_error(
    fit <- tabnet_fit(starwx, starwy, epochs = 1)
  )

  expect_no_error(
    result <- predict(fit, x)
  )


})

test_that("Training hierarchical multi-class for formula", {

  TRUE
})

test_that("Training hierarchical multi-class for data.frame with validation split", {

  expect_no_error(
    fit <- tabnet_fit(starwx, starwy, valid_split = 0.2, epochs = 1)
  )

  expect_no_error(
    result <- predict(fit, starwx, type = "prob")
  )

  expect_equal(ncol(result), 1)

})


test_that("Training hierarchical regression fails with explicit error", {

  TRUE
})

test_that("Training hierarchical multi-class  fails for recipe", {

  TRUE
})
