from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_local_models_are_ignored_but_available_to_container_builds() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert "models/" in gitignore
    assert "models" not in dockerignore
    assert "models/" not in dockerignore

    runtime_dockerignore = (
        ROOT / "docker" / "Dockerfile.vllm-serving.dockerignore"
    ).read_text(encoding="utf-8").splitlines()
    assert "models" in runtime_dockerignore


def test_amphion_spec_bundle_uses_repository_plugin_without_external_clone() -> None:
    dockerfile = (ROOT / "docker" / "Dockerfile.amphion-spec-bundle").read_text(
        encoding="utf-8"
    )

    assert "OPEN_AUDIO_LLM_ENABLE_LEGACY_AMPHION_ASR=1" in dockerfile
    assert "COPY models/amphion-spec /models/amphion-spec" in dockerfile
    assert "git clone" not in dockerfile


def test_kubernetes_profile_contains_only_required_model_services() -> None:
    documents = list(
        yaml.safe_load_all(
            (ROOT / "deploy" / "k8s" / "asr-models.yaml").read_text(encoding="utf-8")
        )
    )
    deployments = {
        document["metadata"]["name"]
        for document in documents
        if document["kind"] == "Deployment"
    }
    services = {
        document["metadata"]["name"]
        for document in documents
        if document["kind"] == "Service"
    }

    assert deployments == {"qwen3-asr", "amphion-spec"}
    assert services == deployments
