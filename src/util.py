from typing import Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn

def fixed_pos_embedding(seq_len: int, dim: int, dtype: jnp.dtype) -> Tuple[jax.Array, jax.Array]:
    """
    Returns:
        sin: (seq_len x dim)
        cos: (seq_len x dim)
    """
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim)[:, None] / dim))
    sinusoid_inp = jnp.dot(jnp.arange(0, seq_len)[:, None], inv_freq.T).astype(dtype)

    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)
    return sin, cos

def apply_rotary_pos_emb(x: jax.Array, sin: jax.Array, cos: jax.Array, scale: jax.Array) -> jax.Array:
    """
    Args:
        x: (batch_size x seq_len x head_size)
        sin: (seq_len x dim)
        cos: (seq_len x dim)
        scale: (seq_len x dim)

    Returns:
        x: (batch_size x seq_len x dim)
    """
    # Double the size of the sin and cos by duplicating each item
    tmp = sin * scale
    sin = jnp.empty((sin.shape[0], sin.shape[1]*2))
    sin = sin.at[:, ::2].set(tmp) # slicing [start:stop:step]
    sin = sin.at[:, 1::2].set(tmp)

    tmp = cos * scale
    cos = jnp.empty((cos.shape[0], cos.shape[1]*2))
    cos = cos.at[:, ::2].set(tmp)
    cos = cos.at[:, 1::2].set(tmp)

    # Rotate every two
    rotated = jnp.stack((-x[:, :, 1::2], x[:, :, ::2]), -1).reshape(x.shape[0], x.shape[1], -1)

    return x * cos + rotated * sin


class XPos(nn.Module):
    head_size: int
    scale_base: int = 512
    dtype: jnp.dtype = jnp.float32
        
    def setup(self):
        self.scale = (jnp.arange(0 ,self.head_size, 2) + 0.4 * self.head_size )/(1.4*self.head_size)
        self.scale = self.scale.astype(self.dtype)

    def __call__(self, x: jax.Array, offset: int = 0, downscale=False) -> jax.Array:
        """
        x: (batch_size x seq_len x head_size)

        Returns: (batch_size x seq_len x head_size)
        """
        length = x.shape[1]
        min_pos = 0
        max_pos = min_pos + offset + length
        scale = self.scale ** ((jnp.arange(min_pos, max_pos)/self.scale_base)[:, None]) # (max_pos x 1)
        sin, cos = fixed_pos_embedding(scale.shape[0], scale.shape[1], self.dtype) # (max_pos x 1), (max_pos x 1)
        # "Cut down" to length
        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:] 
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x

    def forward_reverse(self, x, offset=0, downscale=False) -> jax.Array:
        """
        x: (batch_size x seq_len x head_size)

        Returns: (batch_size x seq_len x head_size)
        """
        length = x.shape[1]
        min_pos = -(length + offset) // 2
        max_pos = min_pos + length + offset
        scale = self.scale ** ((jnp.arange(min_pos, max_pos)/self.scale_base)[:, None])
        sin, cos = fixed_pos_embedding(scale.shape[0], scale.shape[1], self.dtype)
        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]
        if downscale:
            scale = 1 / scale
            
        x = apply_rotary_pos_emb(x, -sin, cos, scale)
        return x
    
