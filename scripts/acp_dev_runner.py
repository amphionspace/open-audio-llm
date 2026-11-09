"""Development job runner for one ACP node: runs shell scripts dropped into an inbox directory.

Each `<name>.sh` placed in ROOT/inbox is moved to ROOT/running, run with bash (cwd
ROOT/running, concurrently with other scripts), its output written to
ROOT/logs/<name>.log and its exit code to ROOT/done/<name>.exit. Scripts choose their
own GPUs via CUDA_VISIBLE_DEVICES. Stop the runner by stopping the ACP job; see
docs/guides/acp.md "单节点开发任务". Submit a copy under /workspace, not this working tree.
usage: acp_dev_runner.py ROOT
"""

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

POLL_SECONDS = 5


def main():
    root = Path(sys.argv[1])
    dirs = {name: root / name for name in ("inbox", "running", "logs", "done")}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    active = {}
    print(f"{datetime.now(timezone.utc):%F %T}Z dev runner on {root}", flush=True)
    while True:
        for script in sorted(dirs["inbox"].glob("*.sh")):
            target = dirs["running"] / script.name
            script.replace(target)
            log = (dirs["logs"] / f"{script.stem}.log").open("w")
            active[script.stem] = (
                subprocess.Popen(
                    ["bash", str(target)],
                    cwd=dirs["running"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                ),
                log,
            )
            print(
                f"{datetime.now(timezone.utc):%F %T}Z start {script.stem}", flush=True
            )
        for name, (process, log) in list(active.items()):
            if process.poll() is not None:
                log.close()
                (dirs["done"] / f"{name}.exit").write_text(f"{process.returncode}\n")
                print(
                    f"{datetime.now(timezone.utc):%F %T}Z exit {name} {process.returncode}",
                    flush=True,
                )
                del active[name]
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
