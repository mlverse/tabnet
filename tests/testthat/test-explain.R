test_that("explain", {

  n <- 10000
  x <- data.frame(
    x = rnorm(n),
    y = rnorm(n),
    z = rnorm(n)
  )
  y <- x$x
  fit <- tabnet_fit(x, y, epochs = 1, verbose = TRUE, num_steps = 3,
                    batch_size = 512)
  fit$fit$importances
  ex <- tabnet_explain(fit, x)
  plot.tabnet_explain(ex)


  data <- resolve_data(x, y = data.frame(rep(1, nrow(x))))

  o <- fit$fit$network$forward_masks(data$x)
  o[[1]] <- sum_embedding_masks(o[[1]], input_dim = fit$fit$network$input_dim, cat_idx = fit$fit$network$cat_idxs, cat_emb_dim = fit$fit$network$cat_emb_dim)
  o[[2]] <- lapply(o[[2]], sum_embedding_masks, cat_idx = fit$fit$network$cat_idxs, cat_emb_dim = fit$fit$network$cat_emb_dim)

  fit$fit$network$cat_idxs
  fit$fit$network$cat_dims





})
