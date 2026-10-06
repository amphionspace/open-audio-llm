"""Checkpoint handoff contract shared by the launcher and the trainer.

Kept free of model runtimes so the CPU launcher can compare it with the schema
AmphionEval reports before training starts.
"""

# Bump when a field changes meaning or is removed; consumers reject unknown versions.
# v2: `checkpoint` may be removed locally once `checkpoint_remote` confirms the upload.
HANDOFF_SCHEMA_VERSION = 2
