# Inspired by:
# https://github.com/Jamie-Stirling/RetNet/blob/main/src/tests.py

import unittest

import jax
import jax.numpy as jnp
from src.model import RetNet
from src.ret_block import GMSRetBlock, RetBlock

class TestRetention(unittest.TestCase):
    def test_simple(self):
        """
        verify that the three implementations of SimpleRetention are identical
        """
        batch_size = 4
        sequence_length = 12
        hidden_size = 6
        chunk_size = 4

        gamma = 0.9
        rand = jax.random.PRNGKey(0)

        X = jax.random.normal(rand, (batch_size, sequence_length, hidden_size))
        sr = RetBlock(hidden_size=hidden_size, head_size=hidden_size, gamma=gamma)
        params = sr.init(rand, X)

        Y_parallel = sr.apply(params, X)

        s_n_1 = jnp.zeros((batch_size, hidden_size, hidden_size))
        Y_recurrent = []
        for i in range(sequence_length):
            y_n, s_n = sr.apply(params, X[:, i:i+1, :], s_n_1, i, method="forward_recurrent")
            Y_recurrent.append(y_n)
            s_n_1 = s_n

        Y_recurrent = jnp.concatenate(Y_recurrent, 1)

        r_n_1 = jnp.zeros((batch_size, hidden_size, hidden_size))
        Y_chunkwise = []
        for i in range(sequence_length // chunk_size):
            y_i, r_i = sr.apply(params, X[:, i*chunk_size:(i+1)*chunk_size, :], r_n_1, i, method="forward_chunkwise")
            Y_chunkwise.append(y_i)
            r_n_1 = r_i
            
        
        Y_chunkwise = jnp.concatenate(Y_chunkwise, 1)

        print(Y_parallel - Y_recurrent)
        assert jnp.allclose(Y_parallel, Y_recurrent, atol=1e-5)
        assert jnp.allclose(Y_parallel, Y_chunkwise, atol=1e-5)
      
  
    def test_multiscale(self):
        """
        verify that the three implementations of MultiScaleRetention are identical
        """
        batch_size = 2
        hidden_size = 6
        sequence_length = 12
        heads = 3
        chunk_size = 2

        rand = jax.random.PRNGKey(0)
        X = jax.random.normal(rand, (batch_size, sequence_length, hidden_size))
        retention = GMSRetBlock(hidden_size, heads)
        params = retention.init(rand, X)

        Y_parallel = retention.apply(params, X)

        s_n_1s = jnp.zeros((batch_size, heads, hidden_size // heads, hidden_size // heads))
        Y_recurrent = []
        for i in range(sequence_length):
            y_n, s_ns = retention.apply(params, X[:, i:i+1, :], s_n_1s, i, method="forward_recurrent")
            Y_recurrent.append(y_n)
            s_n_1s = s_ns

        Y_recurrent = jnp.concatenate(Y_recurrent, 1)

        r_n_1s = jnp.zeros((batch_size, heads, hidden_size // heads, hidden_size // heads))
        Y_chunkwise = []
        for i in range(sequence_length // chunk_size):
            y_i, r_i = retention.apply(params, X[:, i*chunk_size:(i+1)*chunk_size, :], r_n_1s, i, method="forward_chunkwise")
            Y_chunkwise.append(y_i)
            r_n_1s = r_i

        Y_chunkwise = jnp.concatenate(Y_chunkwise, 1)

        self.assertTrue(jnp.allclose(Y_parallel, Y_recurrent, atol=1e-5))
        self.assertTrue(jnp.allclose(Y_parallel, Y_chunkwise, atol=1e-5)) # fails

class TestRetNet(unittest.TestCase):

    def test_retnet(self):
        """
        verify that the three implementations of RetNet are identical
        """
        batch_size = 2
        hidden_size = 36
        sequence_length = 5
        heads = 3
        layers = 4
        ffn_size = 128

        rand = jax.random.PRNGKey(0)
        X = jax.random.normal(rand, (batch_size, sequence_length, hidden_size))
        retnet = RetNet(hidden_size, heads, layers, ffn_size)
        params = retnet.init(rand, X)

        Y_parallel = retnet.apply(params, X)

        s_n_1s = [
            jnp.zeros((batch_size, heads, hidden_size // heads, hidden_size // heads))
            for _ in range(layers)
        ]
        Y_recurrent = []
        for i in range(sequence_length):
            y_n, s_ns = retnet.apply(params, X[:, i:i+1, :], s_n_1s, i, method="forward_recurrent")
            Y_recurrent.append(y_n)
            s_n_1s = s_ns

        Y_recurrent = jnp.concatenate(Y_recurrent, 1)

        r_n_1s = [
            jnp.zeros((batch_size, heads, hidden_size // heads, hidden_size // heads))
            for _ in range(layers)
        ]
        Y_chunkwise = []
        chunk_size = 2
        for i in range(sequence_length // chunk_size):
            y_i, r_i = retnet.apply(params, X[:, i*chunk_size:(i+1)*chunk_size, :], r_n_1s, i, method="forward_chunkwise")

            Y_chunkwise.append(y_i)
            r_n_1s = r_i
        
        Y_chunkwise = jnp.concatenate(Y_chunkwise, 1)

        self.assertTrue(jnp.allclose(Y_parallel, Y_recurrent, atol=1e-5))
        self.assertTrue(jnp.allclose(Y_parallel, Y_chunkwise, atol=1e-5))

if __name__ == "__main__":
    unittest.main()
