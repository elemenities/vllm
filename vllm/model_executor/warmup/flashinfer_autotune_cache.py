# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer autotune cache helpers."""

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import vllm.envs as envs
from vllm.compilation.caching import aot_compile_hash_factors

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def flashinfer_autotune_cache_hash(
    runner: "GPUModelRunner", gpu_topology: Sequence[str] = ()
) -> str:
    """Return a cache key for this model configuration and TP GPU topology."""
    factors = aot_compile_hash_factors(runner.vllm_config)
    cache_key = (factors, tuple(gpu_topology))
    return hashlib.sha256(str(cache_key).encode()).hexdigest()


def resolve_flashinfer_autotune_file(
    runner: "GPUModelRunner", gpu_topology: Sequence[str] = ()
) -> Path:
    override_dir = envs.VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR
    if override_dir:
        root = Path(override_dir).expanduser()
    else:
        from flashinfer.jit import env as flashinfer_jit_env

        flashinfer_workspace = flashinfer_jit_env.FLASHINFER_WORKSPACE_DIR
        root = (
            Path(envs.VLLM_CACHE_ROOT)
            / "flashinfer_autotune_cache"
            / flashinfer_workspace.parent.name
            / flashinfer_workspace.name
        )

    output_dir = root / flashinfer_autotune_cache_hash(runner, gpu_topology)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "autotune_configs.json"


def resolve_flashinfer_autotune_rank_file(cache_path: Path, rank: int) -> Path:
    """Return a rank-private scratch file used to load a shared TP cache."""
    return cache_path.with_name(
        f".{cache_path.stem}.rank-{rank}{cache_path.suffix}"
    )


def prepare_flashinfer_autotune_cache_for_gpu(
    contents: bytes, gpu_name: str
) -> bytes | None:
    """Make shared cache metadata acceptable to one TP rank.

    vLLM deliberately selects one tactic set across all tensor-parallel ranks.
    FlashInfer metadata is local to one GPU, so each rank loads a metadata-
    compatible scratch copy. Other environment metadata remains validated.
    """
    configs = json.loads(contents)
    metadata = configs.get("_metadata")
    if metadata is None:
        return contents
    if not isinstance(metadata, dict) or metadata.get("gpu") == "unknown":
        return None

    configs["_metadata"] = {**metadata, "gpu": gpu_name}
    return json.dumps(configs, separators=(",", ":"), sort_keys=True).encode()


def write_flashinfer_autotune_cache(cache_path: Path, contents: bytes) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=cache_path.parent, suffix=".tmp", prefix=f".{cache_path.name}."
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(contents)
        os.replace(tmp_path, cache_path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_path)
        raise
