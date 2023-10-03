from typing import Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn

from src.util import XPos

class RetBlock(nn.Module):
    hidden_size: int
    head_size: int
    gamma: float

    def setup(self) -> None:
        self.w_q = self.param('w_q', jax.random.normal, (self.hidden_size, self.head_size))
        self.w_k = self.param('w_k', jax.random.normal, (self.hidden_size, self.head_size))
        self.w_v = self.param('w_v', jax.random.normal, (self.hidden_size, self.head_size))
        self.theta = self.param('theta', jax.random.normal, (self.hidden_size, ))
        self.xpos = XPos(self.head_size)
        
    def _create_decay(self, seq_len: int) -> jax.Array:
        """
        Creates a lower triangular matrix of size (batch_size x batch_size)
        D_{n, m} = gamma^{n-m} if n >= m else 0
        """
        m = jnp.arange(seq_len)
        return jnp.tril(self.gamma**(m[:,jnp.newaxis]- m))
     
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        x: (batch_size x seq_len x hidden_size)

        Returns: (batch_size x seq_len x head_size)
        """
        decay_mask = self._create_decay(x.shape[1])
        decay_mask = jnp.expand_dims(decay_mask, 0)

        q = self.xpos(x @ self.w_q) # (batch_size x seq_len x head_size)
        k = self.xpos(x @ self.w_k, downscale=True) # ...
        v = x @ self.w_v # ...

        ret = (q @ k.transpose(0, 2, 1)) * decay_mask # (batch_size x seq_len x seq_len)

        ret = ret @ v # (batch_size x seq_len x head_size)
        return ret
    
    def forward_recurrent(self, x: jax.Array, s_n_1: jax.Array, n: int) -> Tuple[jax.Array, jax.Array]:
        """
        x: (batch_size x 1 x hidden_size)
        s_n_1: (batch_size x head_size x head_size) - previous state
        n: int - current timestep

        Returns: (batch_size x 1 x head_size), (batch_size x head_size x head_size)
        """
        q = self.xpos(x @ self.w_q, n+1) # (batch_size x 1 x head_size)
        k = self.xpos(x @ self.w_k, n+1, downscale=True) # ...
        v = x @ self.w_v # ...


        s_n = self.gamma * s_n_1 + (k.transpose(0, 2, 1) @ v) # (batch_size x head_size x head_size)

        out = q @ s_n # (batch_size x 1 x head_size)

        return out, s_n 
        
    def forward_chunkwise(self, x: jax.Array, s_n_1: jax.Array, n: int) -> Tuple[jax.Array, jax.Array]:
        """
        x: (batch_size x chunk_size x hidden_size)
        s_n_1: (batch_size x head_size x head_size) - previous state

        Returns: (batch_size x chunk_size x head_size), (batch_size x head_size x head_size)
        """
        decay_mask = self._create_decay(x.shape[1])

        q = self.xpos(x @ self.w_q, n*x.shape[1]) # (batch_size x chunk_size x head_size)
        k = self.xpos(x @ self.w_k, n*x.shape[1], downscale=False) # ...
        v = x @ self.w_v # ...
        
        zeta = jnp.expand_dims(decay_mask[-1], (0, 2))    
        s_n = k.transpose(0, 2, 1) @ (v * zeta) + self.gamma**x.shape[1] * s_n_1 # (batch_size x head_size x head_size)
        inner_chunk = (q @ k.transpose(0, 2, 1)) * jnp.expand_dims(decay_mask, 0) # (batch_size x chunk_size x chunk_size)
        inner_chunk = inner_chunk @ v # (batch_size x chunk_size x head_size)

        xi = jnp.repeat(jnp.expand_dims(self.gamma ** jnp.arange(1, x.shape[1]+1), (0, 2)), x.shape[0], 0)
        cross_chunk = (q @ s_n_1) * xi # (batch_size x chunk_size x head_size)
        out = inner_chunk + cross_chunk # (batch_size x chunk_size x head_size)

        return out, s_n


class GMSRetBlock(nn.Module):
    """
    Gated Multiscale Retention
    """
    hidden_size: int
    n_heads: int

    def setup(self) -> None:
        assert self.hidden_size % self.n_heads == 0, "hidden_size must be divisible by heads"
        self.head_size = self.hidden_size // self.n_heads
        self.ret_blocks = [RetBlock(self.hidden_size, self.head_size, 1-2**(-5-i)) for i in range(self.n_heads)]
        self.g_norm = nn.GroupNorm(num_groups=self.n_heads)
        self.w_o = self.param('w_o', jax.random.normal, (self.hidden_size, self.hidden_size))
        self.w_g = self.param('w_g', jax.random.normal, (self.hidden_size, self.hidden_size))

    def _norm_swish(self, x: jax.Array, heads: jax.Array) -> jax.Array:
        """
        The main addition of Gated Multiscale Retention
        -- NOTE: Swish activation is x*sigmoid(x) --
        
        x: (batch_size x seq_len/chunk_size/1 x hidden_size)
        heads: (batch_size x n_heads x head_size)
        """
        y = self.g_norm(heads)
        tmp = x @ self.w_g
        swish = tmp * nn.sigmoid(tmp)

        return (swish * y) @ self.w_o
            

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Parallel version
        x: (batch_size x seq_len x hidden_size)

        Returns: (batch_size x seq_len x hidden_size)
        """
        heads = [ret_block(x) for ret_block in self.ret_blocks]
        heads = jnp.concatenate(heads, axis=2)
        return self._norm_swish(x, heads)


    def forward_recurrent(self, x: jax.Array, s_n_1: jax.Array, n: int) -> Tuple[jax.Array, jax.Array]:
        """
        Recursive Version

        x: (batch_size x 1 x hidden_size)
        s_n_1: (batch_size x n_heads x head_size x head_size) - previous state
        n: int - current step

        Returns: (batch_size x hidden_size), (batch_size x n_heads x head_size x head_size)
        """
        s_n = []
        heads = []
        for i, ret_block in enumerate(self.ret_blocks):
            y_i, s_i = ret_block.forward_recurrent(x, s_n_1[:,i], n)
            heads.append(y_i)
            s_n.append(jnp.expand_dims(s_i, 1))
        heads = jnp.concatenate(heads, axis=2)
        s_n = jnp.concatenate(s_n, axis=1)

        return self._norm_swish(x, heads), s_n


    def forward_chunkwise(self, x: jax.Array, s_n_1: jax.Array, n: int) -> Tuple[jax.Array, jax.Array]:
        """
        Chunkwise Version

        x: (batch_size x chunk_size x hidden_size)
        s_n_1: (batch_size x n_heads x head_size x head_size) - previous state
        n: int - current chunk

        Returns: (batch_size x chunk_size x hidden_size), (batch_size x n_heads x head_size x head_size)
        """
        s_n = []
        y = []
        for i, ret_block in enumerate(self.ret_blocks):
            y_i, s_i = ret_block.forward_chunkwise(x, s_n_1[:,i], n)
            y.append(y_i)
            s_n.append(jnp.expand_dims(s_i, 1))
        y = jnp.concatenate(y, axis=2)
        s_n = jnp.concatenate(s_n, axis=1)
        return self._norm_swish(x, y), s_n


