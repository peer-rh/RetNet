import argparse

import jax
import jax.numpy as jnp
from src.model import RetNet

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_size", type=int, default=2048)
parser.add_argument("--n_heads", type=int, default=16)
parser.add_argument("--n_layers", type=int, default=24)
parser.add_argument("--ffn_size", type=int, default=4096)
parser.add_argument("--mode", type=str, default="par", choices=["par", "rec", "chunk"])
args = parser.parse_args()
if __name__ == "__main__":
    model = RetNet(
        hidden_size=args.hidden_size,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_size=args.ffn_size,
    )
    if args.mode == "par": 
        print(model.tabulate(jax.random.key(0), jnp.ones((64, 128, args.hidden_size))))
    elif args.mode == "rec":
        print(model.tabulate(
            jax.random.key(0),
            jnp.ones((64, 1, args.hidden_size)),
            [jnp.ones((64, args.hidden_size))]*args.n_layers,
            1,
            method="forward_recursive")
        )
    elif args.mode == "chunk":
        print(model.tabulate(
            jax.random.key(0),
            jnp.ones((64, 32, args.hidden_size)),
            [jnp.ones((64, args.hidden_size))]*args.n_layers,
            1,
            method="forward_chunkwise")
        )

