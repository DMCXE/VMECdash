"""VMECdash package.

The package exposes a Dash-free backend surface for VS Code integration while
keeping the existing Dash app available through the optional ``dash`` extra.
"""

import os

# Force JAX onto the CPU platform before it is imported anywhere in the package.
# VMECdash's wout post-processing is lightweight and needs no GPU; a CUDA build of JAX
# would otherwise initialize the GPU and preallocate VRAM on import, which is disruptive
# on shared GPU clusters. Use setdefault so an explicit JAX_PLATFORMS (e.g. "cuda") wins.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

__version__ = "0.1.4"
