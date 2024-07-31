## 3. Implementation Details

### 3.1. Model Details

The details of the model structures and sub-modules are shown in Figure 1. Our encoder-decoder framework mainly contains four parts, the dense feature encoder \( E_F \), the latent token encoder \( E_T \), the implicit motion function (IMF) and the frame decoder \( D_F \), where the IMF is composed of the latent token decoder \( IMF_D \) and implicit motion alignment \( IMF_A \). The ConvLayer [10] block and StyledConv [10] block are directly adopted from the StyleGAN2-pytorch [16] implementation. The \( E_T \) is composed of several ResBlocks [4] and downsample blocks, and a multi-layer perceptron (MLP) is appended to the last, to finally obtain the latent token representation. The latent token decoder \( IMF_D \) is implemented with a StyleGAN2 [10] generator, and the latent token \( t_c \) is injected into the layers using the style modulation operation. For the implicit motion alignment \( IMF_A \) process, it can be formulated as:

$$ 
V' = \text{Attention}(Q, K, V), 
$$
$$ 
= \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V. \tag{5} 
$$

With the aligned values \( V' \), we can further refine them using multi-head self-attention and feed-forward network-based Transformer blocks [19]. In this work, we use 4 stacked transformer decoder blocks.

$$
\text{TransformerBlock}(x) = \text{FFN} \left( \text{MultiHeadSA}(x) \right), 
$$
$$ 
\text{head}_i = \text{Attention}(xW_{Q_i}, xW_{K_i}, xW_{V_i}), 
$$
$$ 
\text{MultiHeadSA}(x) = \text{Cat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h)W_O, 
$$
$$ 
\text{FFN}(x) = \text{GELU} \left( \text{LN}(x)W_1 + b_1 \right) W_2 + b_2, 
$$
$$ 
\text{GELU}(x) = x \cdot \Phi(x), \tag{6} 
$$

where the SA is the self-attention, which takes the output from the previous block, Cat is the concatenation operation, FFN is the feed-forward network, LN is the Layer Normalization [1], GELU [5] is utilized as the activation function and \( \Phi(x) \) is the cumulative distribution function for Gaussian distribution.


```shell


Class ImplicitMotionAlignment:
    Initialize(dim, depth, heads, dim_head, mlp_dim):
        self.cross_attention = CrossAttentionModule(dim, heads, dim_head)
        self.transformer_blocks = [TransformerBlock(dim, heads, dim_head, mlp_dim) for _ in range(depth)]
    
    Forward(ml_c, ml_r, fl_r):
        # Cross-attention module
        V_prime = self.cross_attention(ml_c, ml_r, fl_r)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            V_prime = block(V_prime)
        
        # Reshape to original spatial dimensions
        fl_c = Reshape(V_prime, original_shape)
        
        Return fl_c

        
Class CrossAttentionModule:
    Initialize(dim, heads, dim_head):
        Set up cross-attention parameters and layers
    
    Forward(ml_c, ml_r, fl_r):
        # Flatten inputs
        ml_c = Flatten(ml_c)  # BxCx(HxW)
        ml_r = Flatten(ml_r)  # BxCx(HxW)
        fl_r = Flatten(fl_r)  # BxCx(HxW)
        
        # Add positional embeddings
        ml_c = ml_c + PositionalEmbedding(ml_c)
        ml_r = ml_r + PositionalEmbedding(ml_r)
        
        # Compute attention weights
        attention_weights = Softmax(MatMul(ml_c, Transpose(ml_r)) / sqrt(dim_head))
        
        # Compute output-aligned values
        V_prime = MatMul(attention_weights, fl_r)
        
        Return V_prime

Class TransformerBlock:
    Initialize(dim, heads, dim_head, mlp_dim):
        Set up self-attention and feedforward layers
    
    Forward(x):
        # Self-attention
        x = x + SelfAttention(x)
        
        # Feedforward
        x = x + FeedForward(x)
        
        Return x



# Helper functions (similar to previous pseudocode)
Function Flatten(x):
    Return reshape(x, [B, C, H*W])

Function PositionalEmbedding(x):
    # Implementation of positional embedding (e.g., sinusoidal)
    # ...

Function SelfAttention(x):
    # Implementation of multi-head self-attention
    # ...

Function FeedForward(x):
    # Implementation of feedforward network
    # ...
```