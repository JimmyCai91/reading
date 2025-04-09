#### Reading List 
1. [NLP with Transformers](//C:/Users/caiji/Documents/PDF/Lewis%20Tunstall,%20Leandro%20von%20Werra,%20Thomas%20Wolf%20-%20Natural%20Language%20Processing%20with%20Transformers_%20Building%20Language%20Applications%20with%20Hugging%20Face-O'Reilly%20Media%20(2022).pdf)
2. [2017 Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
3. [2020 An Image Is Worth 16x16 Words](https://arxiv.org/pdf/2010.11929)
4. [2021 Swin Transformer](https://arxiv.org/pdf/2103.14030)


#### Table of Content 
- [Transformer Anatomy](#transformer-anatomy): Encoder, Decoder, MHA, Scaled dot-product Attention, and some of the most prominent architectures

---

#### Transformer Anatomy    
[Table of Content](#table-of-content)  
<div align="center"><img src="../pictures/TransformerAnatomy.png" width=""></div>

---

#### LLM models
- [LLaMA-2](https://cameronrwolfe.substack.com/p/llama-2-from-the-ground-up)


---

**RMSNorm**

[torch.nn.RMSNorm](https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html) | [source](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/normalization.py#L321) | [others](https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py)

$y_i = \frac{x_i}{\mathrm{RMS}(x)} * \gamma_i, \quad
        \text{where} \quad \text{RMS}(x) = \sqrt{\epsilon + \frac{1}{n} \sum_{i=1}^{n} x_i^2}$

```python 

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Normalization
        : param d: model size
        : param p: partial RMSNorm, valid value [0, 1], default -1.0 (disable)
        : param eps: epsilon value, default 1e-8
        : param bias: whether use bias term for RMSNorm, disabled by 
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_paramter("offset", self.offset)
    
    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size
        
        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset
        
        return self.scale * x_normed
```