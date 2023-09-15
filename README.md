# Weighted logsumexp in PyTorch

This is a PyTorch implementation of the `reduce_weighted_logsumexp` function from TensorFlow Probability (see https://www.tensorflow.org/probability/api_docs/python/tfp/math/reduce_weighted_logsumexp). It computes the logarithm of the absolute sum of weighted exponentials of elements across a specified dimension of a tensor in a numerically stable way.


```
output = log(abs(sum(weight * exp(elements across tensor dimensions))))
```



## Function Signature

```python
def weighted_logsumexp(
    logx: torch.Tensor,
    w: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    return_sign: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Computes log(abs(sum(weight * exp(elements across tensor dimensions)))) in a numerically stable way.
    
    :param logx: Tensor to reduce
    :param w: Weights
    :param dim: Dimension to reduce
    :param keep_dim: If True, retains reduced dimensions with length 1
    :param return_sign: If True, return the sign of weight * exp(elements across tensor dimensions)
    
    :return: Either the reduced tensor or a tuple of the reduced tensor and the sign
    """
```

## Usage
    
```python
import torch
from torch_weighted_logsumexp.weighted_logsumexp import weighted_logsumexp

# Create an example tensor and weights
logx = torch.tensor([0.0, 0.0, 0.0])

w = torch.tensor([-1.0, 1.0, 1.0])

print(weighted_logsumexp(logx, w, dim=0))
```

## Limitations
- The dimension to reduce must be specified as an integer, unlike usual PyTorch reduction functions it is not possible to select multiple dimensions at once to reduce.

## Requirements
- PyTorch 1.7.0 or higher, Tested with PyTorch 2.0.0 and 2.0.1