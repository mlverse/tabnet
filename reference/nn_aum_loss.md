# AUM loss

Creates a criterion that measures the Area under the \\Min(FPR, FNR)\\
(AUM) between each element in the input \\pred_tensor\\ and target
\\label_tensor\\.

## Usage

``` r
nn_aum_loss()
```

## Details

This is used for measuring the error of a binary reconstruction within
highly unbalanced dataset, where the goal is optimizing the ROC curve.
Note that the targets \\label_tensor\\ should be factor level of the
binary outcome, i.e. with values `1L` and `2L`.

## Examples

``` r
loss <- nn_aum_loss()
input <- torch::torch_randn(4, 6, requires_grad = TRUE)
target <- input > 1.5
output <- loss(input, target)
output$backward()
```
