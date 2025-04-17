test_that("nn_aum_loss works as expected", {
  
  aum_loss <- tabnet:::nn_aum_loss()
  
  # the poor-guy expect_r6_class(x, class)
  expect_true(all(c("nn_mse_loss","nn_loss","nn_module") %in% class(aum_loss)))
  
  label_tensor <- torch::torch_tensor(attrition$Attrition)
  pred_tensor <- torch::torch_rand(label_tensor$shape, requires_grad = TRUE) 
  output <- aum_loss(pred_tensor, label_tensor)
  output$backward()
  
  expect_tensor(output)
})


test_that("nn_unsupervised_loss is working as expected", {
  
  unsup_loss <- tabnet:::nn_unsupervised_loss()
  
  # the poor-guy expect_r6_class(x, class)
  expect_true(all(c("nn_weighted_loss","nn_loss","nn_module") %in% class(unsup_loss)))
  
  y_pred <- torch::torch_rand(3,5, requires_grad = TRUE)
  embedded_x <- torch::torch_rand(3,5)
  obfuscation_mask <- torch::torch_bernoulli(embedded_x, p = 0.5)
  output <- unsup_loss(y_pred, embedded_x, obfuscation_mask)
  output$backward()
  
  expect_tensor(output)
})



