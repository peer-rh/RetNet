from typing import List, Tuple
import jax
import flax.linen as nn
from src.ret_block import GMSRetBlock

class RetNet(nn.Module):
    hidden_size: int
    n_heads: int
    n_layers: int
    ffn_size: int

    def setup(self) -> None:
        self.ret_blocks = [GMSRetBlock(self.hidden_size, self.n_heads) for _ in range(self.n_layers)]
        self.ffn_s = [
                nn.Sequential([
                    nn.Dense(self.ffn_size, use_bias=False),
                    nn.gelu,
                    nn.Dense(self.hidden_size, use_bias=False),
                ]) for _ in range(self.n_layers)
            ]

        self.layer_norms_1 = [nn.LayerNorm() for _ in range(self.n_layers)]
        self.layer_norms_2 = [nn.LayerNorm() for _ in range(self.n_layers)]
        return super().setup()

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        x: (batch_size x seq_len x hidden_size)

        Returns: (batch_size x seq_len x hidden_size)
        """
        for i in range(self.n_layers):
            x = self.ret_blocks[i].forward_par(self.layer_norms_1[i](x)) + x
            x = self.ffn_s[i](self.layer_norms_2[i](x)) + x
        return x

    def forward_recursive(self, x: jax.Array, s_n_1: List[jax.Array], n: int) -> Tuple[jax.Array, List[jax.Array]]:
        """
        x: (batch_size x hidden_size)
        s_n_1: [(batch_size x hidden_size) x n_layers] - previous state

        Returns: (batch_size x hidden_size) x [(batch_size x hidden_size) x n_layers]
        """
        s_n = []
        for i in range(self.n_layers):
            y, s_i = self.ret_blocks[i].forward_rec(self.layer_norms_1[i](x), s_n_1[i], n)
            s_n.append(s_i)
            x = y + x
            x = self.ffn_s[i](self.layer_norms_2[i](x)) + x
        return x, s_n

    def forward_chunkwise(self, x: jax.Array, s_n_1: List[jax.Array], n: int) -> Tuple[jax.Array, List[jax.Array]]:
        """
        x: (batch_size x chunk_size x hidden_size)
        s_n_1: [(batch_size x hidden_size) x n_layers] - previous state

        Returns: (batch_size x chunk_size x hidden_size) x [(batch_size x hidden_size) x n_layers]
        """
        s_n = []
        for i in range(self.n_layers):
            y, s_i = self.ret_blocks[i].forward_chunk(self.layer_norms_1[i](x), s_n_1[i], n)
            s_n.append(s_i)
            x = y + x
            x = self.ffn_s[i](self.layer_norms_2[i](x)) + x
        return x, s_n

