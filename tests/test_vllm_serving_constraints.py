from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONSTRAINTS = ROOT / "constraints" / "vllm-serving.txt"


def _pins() -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw_line in CONSTRAINTS.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        assert "[" not in line and "]" not in line
        assert "==" in line
        name, version = line.split("==", 1)
        assert name
        assert version
        assert not any(op in version for op in ("<", ">", "~=", "!="))
        pins[name.lower()] = version
    return pins


def test_vllm_serving_constraints_pin_verified_runtime():
    pins = _pins()

    assert pins["vllm"] == "0.18.0"
    assert pins["torch"] == "2.10.0"
    assert pins["torchaudio"] == "2.10.0"
    assert pins["torchvision"] == "0.25.0"
    assert pins["transformers"] == "4.57.6"
    assert pins["triton"] == "3.6.0"
    assert pins["tritonclient"] == "2.70.0"


def test_vllm_serving_constraints_do_not_contain_loose_runtime_inputs():
    pins = _pins()

    assert "ray" not in pins
    assert all(">" not in value and "<" not in value for value in pins.values())


def test_vllm_serving_dockerfile_documents_runtime_compiler_boundary():
    dockerfile = (ROOT / "docker" / "Dockerfile.vllm-serving").read_text(
        encoding="utf-8"
    )

    assert "build-essential" in dockerfile
    assert "constraints/vllm-serving.txt" in dockerfile
    assert "qwen3_asr.py" in dockerfile


def test_vllm_compose_profile_enables_embedding_bypass():
    compose = (ROOT / "compose.vllm.yaml").read_text(encoding="utf-8")

    assert "OPEN_AUDIO_LLM_MODEL" in compose
    assert "- -e" in compose
    assert "- -q" in compose
