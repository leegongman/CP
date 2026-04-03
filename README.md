# CP Checkpoints

This repo stores OmniQuant checkpoints in a clone-friendly split format.

- Each checkpoint directory contains all small files directly.
- `model.safetensors` is split into `model.safetensors.part-*`.
- Run `./reconstruct_all.sh` after clone to rebuild every `model.safetensors`.
