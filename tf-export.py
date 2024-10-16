import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer
import torch
from model import IMFModel
from nobuco import converter, ChannelOrderingStrategy
import tensorflow as tf
import torch.nn as nn

import tensorflow as tf

class CustomMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.scaling = self.head_dim ** -0.5
        
        self.in_proj_weight = self.add_weight("in_proj_weight", shape=(embed_dim, 3 * embed_dim),
                                              initializer="glorot_uniform")
        if bias:
            self.in_proj_bias = self.add_weight("in_proj_bias", shape=(3 * embed_dim,),
                                                initializer="zeros")
        else:
            self.in_proj_bias = None
        
        self.out_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias)

    def _in_projection_packed(self, q, k, v):
        w = self.in_proj_weight
        b = self.in_proj_bias if self.in_proj_bias is not None else None
        E = q.shape[-1]
        
        # Debug print statement for input shapes of q, k, v
        tf.print("Shapes of q, k, v before projection:", tf.shape(q), tf.shape(k), tf.shape(v))
        
        if k is v:
            if q is k:
                # self-attention
                proj = tf.einsum("bne,ef->bnf", q, w) + b
                q, k, v = tf.split(proj, 3, axis=-1)
            else:
                # encoder-decoder attention
                w_q, w_kv = tf.split(w, [E, E * 2])
                if b is not None:
                    b_q, b_kv = tf.split(b, [E, E * 2])
                else:
                    b_q = b_kv = None
                q = tf.einsum("bne,ef->bnf", q, w_q) + b_q
                kv = tf.einsum("bne,ef->bnf", k, w_kv) + b_kv
                k, v = tf.split(kv, 2, axis=-1)
        else:
            w_q, w_k, w_v = tf.split(w, 3)
            if b is not None:
                b_q, b_k, b_v = tf.split(b, 3)
            else:
                b_q = b_k = b_v = None
            q = tf.einsum("bne,ef->bnf", q, w_q) + b_q
            k = tf.einsum("bne,ef->bnf", k, w_k) + b_k
            v = tf.einsum("bne,ef->bnf", v, w_v) + b_v
        
        # Debug print statement after projection
        tf.print("Shapes of q, k, v after projection:", tf.shape(q), tf.shape(k), tf.shape(v))
        return q, k, v

    def call(self, query, key, value, need_weights=False, attn_mask=None):
        tgt_len, bsz, embed_dim = query.shape
        
        # Debug print statement for input shapes
        tf.print("Input query shape:", tf.shape(query))
        tf.print("Input key shape:", tf.shape(key))
        tf.print("Input value shape:", tf.shape(value))
        
        q, k, v = self._in_projection_packed(query, key, value)
        
        # Debug print for shapes of q, k, v after reshaping
        tf.print("q, k, v shapes after projection and before reshape:", tf.shape(q), tf.shape(k), tf.shape(v))
        
        q = tf.reshape(q, (tgt_len, bsz * self.num_heads, self.head_dim))
        k = tf.reshape(k, (tgt_len, bsz * self.num_heads, self.head_dim))
        v = tf.reshape(v, (tgt_len, bsz * self.num_heads, self.head_dim))
        
        # Debug print for shapes of q, k, v after reshaping
        tf.print("q, k, v shapes after reshape:", tf.shape(q), tf.shape(k), tf.shape(v))
        
        q = tf.transpose(q, [1, 0, 2])
        k = tf.transpose(k, [1, 0, 2])
        v = tf.transpose(v, [1, 0, 2])
        
        # Debug print after transposition
        tf.print("q, k, v shapes after transpose:", tf.shape(q), tf.shape(k), tf.shape(v))
        
        attn_output_weights = tf.einsum('bie,bje->bij', q, k)
        attn_output_weights = attn_output_weights * self.scaling
        
        # Debug print for attention weights
        tf.print("Attention output weights shape:", tf.shape(attn_output_weights))
        
        if attn_mask is not None:
            attn_output_weights += attn_mask
        
        attn_output_weights = tf.nn.softmax(attn_output_weights, axis=-1)
        attn_output_weights = tf.nn.dropout(attn_output_weights, self.dropout)
        
        attn_output = tf.einsum('bij,bje->bie', attn_output_weights, v)
        
        # Debug print for attention output
        tf.print("Attention output shape before final projection:", tf.shape(attn_output))
        
        attn_output = tf.reshape(attn_output, (bsz, tgt_len, embed_dim))
        attn_output = self.out_proj(attn_output)
        
        # Debug print for final output
        tf.print("Final attention output shape:", tf.shape(attn_output))
        
        if need_weights:
            attn_output_weights = tf.reshape(attn_output_weights, (bsz, self.num_heads, tgt_len, tgt_len))
            attn_output_weights = tf.reduce_mean(attn_output_weights, axis=1)
            return attn_output, attn_output_weights
        else:
            return attn_output


@nobuco.converter(nn.MultiheadAttention, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_MultiheadAttention(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
    custom_mha = CustomMultiHeadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=self.dropout)
    
    # Transpose and reshape the weights to match TensorFlow's expected shape
    in_proj_weight = self.in_proj_weight.detach().numpy()
    custom_mha.in_proj_weight.assign(in_proj_weight)
    
    if self.in_proj_bias is not None:
        custom_mha.in_proj_bias.assign(self.in_proj_bias.detach().numpy())
    
    out_proj_weight = self.out_proj.weight.detach().numpy().T
    custom_mha.out_proj.kernel.assign(out_proj_weight)
    
    if self.out_proj.bias is not None:
        custom_mha.out_proj.bias.assign(self.out_proj.bias.detach().numpy())
    
    def func(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        # Transpose inputs to match TensorFlow's expected shape
        query = tf.transpose(query, [1, 0, 2])
        key = tf.transpose(key, [1, 0, 2])
        value = tf.transpose(value, [1, 0, 2])
        
        output = custom_mha(query, key, value, need_weights=need_weights, attn_mask=attn_mask)
        
        if need_weights:
            attn_output, attn_weights = output
            # Transpose output back to PyTorch's expected shape
            attn_output = tf.transpose(attn_output, [1, 0, 2])
            return attn_output, attn_weights
        else:
            # Transpose output back to PyTorch's expected shape
            return tf.transpose(output, [1, 0, 2])
    
    return func

@converter(torch.flip, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_flip(input: torch.Tensor, dims):
    def func(input, dims):
        # Ensure dims is a tensor
        dims = tf.convert_to_tensor(dims, dtype=tf.int32)
        
        # Convert PyTorch-style negative indices to TensorFlow-style
        rank = tf.rank(input)
        dims = tf.where(dims < 0, dims + rank, dims)
        
        # TensorFlow's reverse operation expects a list of axes to reverse
        # We don't need to create a boolean mask anymore
        return tf.reverse(input, dims)
    return func
if __name__ == "__main__":
    # Load your PyTorch model
    pytorch_model = IMFModel()
    pytorch_model.eval()

    # Load the checkpoint
    checkpoint = torch.load("./checkpoints/checkpoint.pth", map_location='cpu')
    pytorch_model.load_state_dict(checkpoint['model_state_dict'])

    x_current = torch.randn(1, 3, 256, 256)
    x_reference = torch.randn(1, 3, 256, 256)
    keras_model = nobuco.pytorch_to_keras(
        pytorch_model,
        args=[x_current,x_reference], kwargs=None,
        inputs_channel_order=ChannelOrder.PYTORCH,
        outputs_channel_order=ChannelOrder.TENSORFLOW
    )

    # tfjs.converters.save_keras_model(keras_model, 'keras')
    keras_model.save("imf/imf.keras")

