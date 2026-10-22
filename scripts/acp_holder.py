"""Hold ACP capacity for this project: keep NODES nodes busy with ASR- jobs (AGENTS.md "ACP Compute").

Polls the pool; when the ASR- jobs hold fewer than --nodes nodes and a whole node is idle, submits one
GPU-burning holder job (scripts/gpu_hold.py) (display name ASR-<--name-stem>-h<HHMM>, an ordinary experiment-style name) and exits. Real or debug jobs are queued first and the holder
is stopped afterwards (`sco acp jobs stop`). Cluster settings (image, spec, mounts) come from the `cluster:` section
of any train config. Usage: acp_holder.py --config <train.yaml> [--nodes 2] [--interval 120] [--once]
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import yaml

from open_audio_llm.cluster import JOB_PREFIX, refresh_sco, sco_environment

GPUS_PER_NODE = 8


def run_sco(cluster, *args):
    process = subprocess.run([f"{cluster['sco_home']}/bin/sco", *args], env=sco_environment(cluster),
                             capture_output=True, text=True, timeout=120, check=False)
    return process.returncode, process.stdout + process.stderr


def asr_nodes(table):
    """Nodes held by ASR- jobs in `sco acp jobs list` output (a long display name wraps onto following rows)."""
    total = 0
    for line in table.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 8 and cells[0].startswith("pt-") and cells[2].startswith(JOB_PREFIX):
            total += int(cells[6])
    return total


def idle_gpus(usage):
    for line in usage.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == "GPU_NUMBER":
            return int(cells[3])
    raise ValueError("GPU_NUMBER row not found in cluster usage")


def holder_command(cluster, python):
    # A fixed copy under /workspace: the job must not depend on which branch the working tree has checked out.
    target = Path(cluster["sco_home"]).parent / "asr-holder/gpu_hold.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).with_name("gpu_hold.py"), target)
    return f"{python} {target}"


def submit_holder(cluster, python, stem):
    name = f"{JOB_PREFIX}{stem}-h{datetime.now(timezone.utc):%H%M}"
    code, output = run_sco(
        cluster, "acp", "jobs", "create", f"--workspace-name={cluster['workspace']}",
        f"--aec2-name={cluster['aec2']}", f"--job-name={name}",
        f"--container-image-url={cluster['image']}", "--training-framework=pytorch",
        f"--worker-spec={cluster['worker_spec']}", "--worker-nodes=1", "--priority=NORMAL",
        f"--quota-type={cluster.get('quota_type', 'reserved')}",
        f"--storage-mount={','.join(cluster['mounts'])}", f"--command={holder_command(cluster, python)}")
    if code:
        raise RuntimeError(f"holder submission failed: {output[-500:]}")
    return name, output.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--name-stem", default="ts-robust-v7set-continue-from016-ckpt15000-1node")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))
    cluster, python = config["cluster"], config["runtime"]["python"]
    while True:
        refresh_sco(cluster)
        base = ["acp", "jobs", "list", f"--workspace-name={cluster['workspace']}", f"--aec2-name={cluster['aec2']}", "--state=RUNNING"]
        held = asr_nodes(run_sco(cluster, *base)[1])
        idle = idle_gpus(run_sco(cluster, "aec2", "clusters", "usage", f"--name={cluster['aec2']}")[1])
        print(f"{datetime.now(timezone.utc):%F %T}Z ASR nodes {held}/{args.nodes}, idle GPUs {idle}", flush=True)
        if held < args.nodes and idle >= GPUS_PER_NODE:
            print(*submit_holder(cluster, python, args.name_stem), flush=True)
            return 0
        if args.once:
            return 1
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
