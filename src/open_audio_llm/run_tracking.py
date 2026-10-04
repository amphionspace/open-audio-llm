"""CPU-only W&B recorder; readiness means remotely acknowledged metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import yaml

from .cli import write_json


def numeric_tree(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(numeric_tree(item, f"{prefix}/{key}" if prefix else key))
    elif isinstance(value, (int, float)):
        import math

        if math.isfinite(value):
            result[prefix] = value
    return result


def collect_metrics(config):
    events = {}
    task = config["task"]
    output = task.get("arguments", {}).get("--output_dir")
    if task["kind"] == "train" and output:
        from .scripts.sync_wandb import collect_events

        if Path(output).name != "training":
            raise ValueError("Tracked training output must end in /training")
        events.update(collect_events(Path(output).parent))
    for source in config["tracking"].get("metrics", []):
        path = Path(source["file"])
        if not path.exists():
            continue
        if source.get("format") == "jsonl":
            rows = [
                json.loads(line)
                for line in path.read_bytes().splitlines(keepends=True)
                if line.endswith(b"\n")
            ]
        else:
            try:
                rows = [json.loads(path.read_text())]
            except json.JSONDecodeError:
                continue
        for i, row in enumerate(rows):
            values = numeric_tree(row, source.get("prefix", "result"))
            key = source.get("prefix", "result") + ":" + str(i)
            events[key] = values
    return events


def verify(wandb, run, expected, timeout, terminal=None):
    deadline = time.monotonic() + timeout
    while True:
        remote = wandb.Api().run(f"{run.entity}/{run.project}/{run.id}")
        actual = dict(remote.summary)
        # W&B's JSON round trip can change the final bits of a float.
        matched = all(
            actual.get(k) == v
            or (
                isinstance(v, float)
                and isinstance(actual.get(k), (int, float))
                and math.isclose(actual[k], v, rel_tol=1e-14, abs_tol=0)
            )
            for k, v in expected.items()
        )
        if matched and (
            terminal is None or remote.state == terminal
        ):
            return {
                "url": remote.url,
                "state": remote.state,
                "verified_metrics": expected,
                "verified_unix": time.time(),
            }
        if time.monotonic() >= deadline:
            raise RuntimeError("W&B remote metrics or terminal state did not match")
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    options = parser.parse_args()
    attempt = options.attempt
    config = yaml.safe_load((attempt / "effective.yaml").read_text())
    tracking = config["tracking"]
    import wandb

    run = wandb.init(
        entity=tracking["entity"],
        project=tracking["project"],
        id=tracking.get("run_id") or f"{tracking['name']}-{attempt.name}",
        name=f"{tracking['name']}-{attempt.name}",
        resume="allow",
        mode="online",
        config=config,
        dir=str(attempt),
        settings=wandb.Settings(
            console="off",
            disable_git=True,
            disable_code=True,
            save_code=False,
            x_disable_stats=True,
            x_disable_meta=True,
        ),
    )
    expected = {"launcher/started": 1, "launcher/attempt": str(attempt)}
    run.log(expected)
    run.summary.update(expected)
    write_json(
        attempt / "wandb-start-verification.json",
        verify(wandb, run, expected, tracking.get("startup_timeout", 180)),
    )
    run.define_metric("global_step")
    for prefix in ("train/*", "validation/*", "eval/*", "perf/*"):
        run.define_metric(prefix, step_metric="global_step")
    remote = wandb.Api().run(f"{run.entity}/{run.project}/{run.id}")
    seen = {
        row["sync_event"]
        for row in remote.scan_history(keys=["sync_event"])
        if "sync_event" in row
    }
    try:
        while True:
            events = collect_metrics(config)
            for key, values in sorted(
                events.items(),
                key=lambda pair: (pair[1].get("global_step", 0), pair[0]),
            ):
                signature = (
                    key
                    + ":"
                    + hashlib.sha256(
                        json.dumps(values, sort_keys=True).encode()
                    ).hexdigest()
                )
                if signature not in seen:
                    run.log({"sync_event": signature, **values})
                    run.summary.update(values)
                    expected.update(values)
                    seen.add(signature)
            stop = attempt / "tracking-stop.json"
            if stop.exists():
                code = json.loads(stop.read_text())["exit_code"]
                break
            time.sleep(tracking.get("interval", 5))
        expected.update(
            {"launcher/exit_code": code, "launcher/metric_events": len(seen)}
        )
        run.summary.update(expected)
        run.finish(exit_code=code)
        terminal = "finished" if code == 0 else "failed"
        write_json(
            attempt / "wandb-verification.json",
            verify(wandb, run, expected, tracking.get("finish_timeout", 90), terminal),
        )
    except BaseException:
        run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
