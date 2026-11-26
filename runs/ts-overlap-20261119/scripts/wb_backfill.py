"""Backfill the history rows a crashed online W&B run never uploaded, from its local .wandb transaction log.
Reads the leveldb-log framing (7-byte file header, 32 KiB blocks, 7-byte record headers, FULL/FIRST/MIDDLE/LAST
chunks), decodes Record protobufs, keeps history rows with _step > the remote max, then resumes the run and logs them
with their original steps.  usage: wb_backfill.py RUN_WANDB_FILE ENTITY/PROJECT/RUN_ID [--summary=wandb-summary.json] [--dry-run]"""
import json, struct, sys
from wandb.proto import wandb_internal_pb2 as pb

BLOCK = 32768
FULL, FIRST, MIDDLE, LAST = 1, 2, 3, 4


def records(path):
    data = open(path, "rb").read()
    ident, magic, version = struct.unpack("<4sHB", data[:7])
    assert ident == b":W&B" and magic == 0xBEE1, (ident, magic, version)
    pos, buf = 7, b""
    while pos + 7 <= len(data):
        left = BLOCK - pos % BLOCK
        if left < 7:
            pos += left
            continue
        _crc, length, kind = struct.unpack("<IHB", data[pos:pos + 7])
        if kind == 0 and length == 0:  # zero padding at the end of a block
            pos += left
            continue
        chunk = data[pos + 7:pos + 7 + length]
        if len(chunk) < length:
            return  # truncated tail (process killed mid-write)
        pos += 7 + length
        if kind == FULL:
            yield chunk
        elif kind == FIRST:
            buf = chunk
        elif kind == MIDDLE:
            buf += chunk
        elif kind == LAST:
            yield buf + chunk
            buf = b""


def history_rows(path):
    for raw in records(path):
        r = pb.Record()
        r.ParseFromString(raw)
        if r.WhichOneof("record_type") != "history":
            continue
        row = {}
        for it in r.history.item:
            key = it.key or "/".join(it.nested_key)
            row[key] = json.loads(it.value_json)
        yield row


def main():
    path, run_path = sys.argv[1], sys.argv[2]
    dry = "--dry-run" in sys.argv
    rows = list(history_rows(path))
    summary = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--summary=")), None)
    if summary:  # the final row can sit in the truncated tail; the summary file holds that step's values
        last = json.load(open(summary))
        if last.get("_step", -1) > rows[-1]["_step"]:
            rows.append(last)
    steps = [r.get("_step") for r in rows]
    print(f"local history rows {len(rows)}, steps {steps[0]}..{steps[-1]}")
    import wandb
    api_run = wandb.Api(timeout=120).run(run_path)
    remote_max = api_run.summary.get("_step")
    todo = [r for r in rows if r.get("_step") is not None and r["_step"] > remote_max]
    print(f"remote max _step {remote_max}; rows to upload {len(todo)}"
          + (f" ({todo[0]['_step']}..{todo[-1]['_step']})" if todo else ""))
    if dry or not todo:
        if todo:
            print("first row keys", sorted(k for k in todo[0] if not k.startswith("_")))
        return
    entity, project, run_id = run_path.split("/")
    run = wandb.init(entity=entity, project=project, id=run_id, resume="must")
    for r in todo:
        step = r["_step"]
        run.log({k: v for k, v in r.items() if not k.startswith("_")}, step=step)
    run.finish()
    print("uploaded", len(todo))


main()
