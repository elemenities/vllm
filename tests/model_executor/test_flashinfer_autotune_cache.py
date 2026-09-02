# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

from vllm.model_executor.warmup import flashinfer_autotune_cache as cache


def test_cache_hash_includes_ordered_gpu_topology(monkeypatch):
    monkeypatch.setattr(cache, "aot_compile_hash_factors", lambda _: ("config",))
    runner = SimpleNamespace(vllm_config=object())

    assert cache.flashinfer_autotune_cache_hash(
        runner, ("RTX 5080", "RTX 5060 Ti")
    ) != cache.flashinfer_autotune_cache_hash(
        runner, ("RTX 5060 Ti", "RTX 5080")
    )


def test_shared_cache_is_relabelled_only_for_local_gpu():
    contents = json.dumps(
        {
            "_metadata": {
                "gpu": "RTX 5080",
                "cuda_version": "13.0",
                "flashinfer_version": "0.6.18",
            },
            "tactic": {"runner": "plan"},
        }
    ).encode()

    prepared = cache.prepare_flashinfer_autotune_cache_for_gpu(
        contents, "RTX 5060 Ti"
    )

    assert prepared is not None
    result = json.loads(prepared)
    assert result["_metadata"] == {
        "gpu": "RTX 5060 Ti",
        "cuda_version": "13.0",
        "flashinfer_version": "0.6.18",
    }
    assert result["tactic"] == {"runner": "plan"}


def test_indeterminate_shared_cache_is_not_loaded():
    contents = json.dumps({"_metadata": {"gpu": "unknown"}}).encode()

    assert (
        cache.prepare_flashinfer_autotune_cache_for_gpu(contents, "RTX 5060 Ti")
        is None
    )
