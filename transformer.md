#### Reading List 

**Paper**

1. [NLP with Transformers](//C:/Users/caiji/Documents/PDF/Lewis%20Tunstall,%20Leandro%20von%20Werra,%20Thomas%20Wolf%20-%20Natural%20Language%20Processing%20with%20Transformers_%20Building%20Language%20Applications%20with%20Hugging%20Face-O'Reilly%20Media%20(2022).pdf)
2. [2017 Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
3. [2020 An Image Is Worth 16x16 Words](https://arxiv.org/pdf/2010.11929)
4. [2021 Swin Transformer](https://arxiv.org/pdf/2103.14030)


**Article**

1. [What is RAG?](https://cohere.com/blog/what-is-rag)
2. [From Basic to Advanced RAG every step of the way](https://rahuld3eora.medium.com/from-basic-to-advanced-rag-every-step-of-the-way-dee3a3a1aae9)
3. [Open LLMs](https://github.com/eugeneyan/open-llms?tab=readme-ov-file)

**Code**  

1. [Hands on LLMs](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/tree/main)  
2. [LlamaIndex](https://github.com/run-llama/llama_index/tree/main): [Question-Answering](https://docs.llamaindex.ai/en/stable/use_cases/q_and_a/), [BM25 Retriever](https://docs.llamaindex.ai/en/stable/examples/retrievers/bm25_retriever/)  


---

#### Table of Content 

- [Transformer Anatomy](#transformer-anatomy): Encoder, Decoder, MHA, Scaled dot-product Attention, and some of the most prominent architectures

---

#### LLMs

- [BERT](https://cameronrwolfe.substack.com/p/language-understanding-with-bert#§berts-architecture)  
- [PaLM](https://blog.eleuther.ai/rotary-embeddings/)  
- [LLaMA-2](https://cameronrwolfe.substack.com/p/llama-2-from-the-ground-up)  
- [GQA - explained with code](https://medium.com/@maxshapp/grouped-query-attention-gqa-explained-with-code-e56ee2a1df5a)
- [Qwen1.5-110B](https://qwenlm.github.io/zh/blog/qwen1.5-110b/)


---

**Multi-Headed, Causal Self-Attention**  

$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$  
$MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^o, \text{where } head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)$

**Grouped-query Attention**  

```python  
import torch
import torch.nn.functional as F
from einops import einsum, rearrange

# shapes: (batch_size, seq_len, num_heads, head_dim)  
query = torch.randn(1, 256, 8, 64)  
key = torch.randn(1, 256, 2, 64)  
value = torch.randn(1, 256, 2, 64)  

# define number of heads in one group, in this toy example we have 2 kv_heads, 
# so this means we will have 2 groups of size 4 each
num_head_groups = query.shape[2] // key.shape[2]
scale = query.size(-1) ** 0.5

# Swap seq_len with num_heads to accelerate computations
query = rearrange(query, "b n h d -> b h n d")
key = rearrange(key, "b s h d -> b h s d")
value = rearrange(value, "b s h d -> b h s d")

# split query num_heads in groups by introducing additional `g` dimension
query = rearrange(query, "b (h g) n d -> b g h n d", g=num_head_groups)

# calculate the attention scores and sum over the group dim to perform averaging 
scores = einsum(query, key, "b g h n d, b h s d -> b h n s")
attention = F.softmax(scores / scale, dim=-1)

# apply weights to the value head
out = einsum(attention, value, "b h n s, b h s d -> b h n d")

# reshape back to original dimensions
out = rearrange(out, "b h n d -> b n h d")
```



---

**Swish Activation**

[torch.nn.SiLU](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html) | [Link](https://medium.com/@jiangmen28/beyond-relu-discovering-the-power-of-swiglu-超越-relu-发现-swiglu-的力量-9dbc7d8258bf)

$\text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}$

```python
import torch 
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        # Adjust hidden_dim to be a multiple of multiple_of
        hidden_dim = multiple_of * ((2 * hidden_dim // 3 + multiple_of -1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim)    # First linear transformation
        self.w2 = nn.Linear(hidden_dim, dim)    # Second linear transformation
        self.w3 = nn.Linear(dim, hidden_dim)    # Third linear transformation 
        self.dropout = nn.Dropout(dropout)      # Dropout layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass using Swish activation and dropout
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
```

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

---

**Positional Embedding**

[Link](https://huggingface.co/blog/designing-positional-encoding)