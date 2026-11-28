# Rewrite the wenetspeech_clean description include list (ids carry "dataset@version:split:") to final v4 and count hits
# in v4's record index (README "并入阶段 2 的规格"). usage: wenet_v4_include.py <out.txt>
import sys
from pathlib import Path
import lance, pyarrow.compute as pc
OLD = "train-punc-v1-20261102-final-v1-20261120"; NEW = "train-punc-v1-20261102-final-v4-20261127"
src = Path("/workspace/data/datasets/context_correction_clean_zh/versions/think-v2-20261113-final-v1-20261120/wenetspeech_clean/include-wenetspeech_clean_train-punc-v1-20261102-final-v1-20261120-train.txt")
out = Path(sys.argv[1])
pre_old, pre_new = f"wenetspeech_clean@{OLD}:", f"wenetspeech_clean@{NEW}:"
assert all(i[len(pre_old):].startswith("train:") for i in src.read_text().split())
ids = src.read_text().split()
assert all(i.startswith(pre_old) for i in ids), "unexpected prefix"
new = [pre_new + i[len(pre_old):] for i in ids]
assert len(set(new)) == len(new) == 923
d = lance.dataset(f"/workspace/data/datasets/wenetspeech_clean/versions/{NEW}/record_index/table.lance")
t = d.to_table(columns=["id", "duration"], filter=pc.field("id").isin([i[len(pre_new) + len("train:"):] for i in new]))
print("hits", t.num_rows, "of", len(new), "max_duration", max(t.column("duration").to_pylist()))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(new) + "\n")
print("wrote", out)
