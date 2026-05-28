import os

if os.environ.get("SEQMASK_BACKEND", "torch") == "torch":
    from seqmask.backend import torch_backend as backend

    print("Using PyTorch backend for seqmask.")
elif os.environ.get("SEQMASK_BACKEND") in ["JAX", "jax"]:
    from seqmask.backend import jax_backend as backend

    print("Using JAX backend for seqmask.")
else:
    raise ValueError(
        f"Unsupported SEQMASK_BACKEND: {os.environ.get('SEQMASK_BACKEND')}"
    )
