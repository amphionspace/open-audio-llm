"""CPU check of the decision-batch path of ctx_kl_train.py (README "后续 3"), before training:
- the decision stream (second replay-data instance, seed decision_seed) is not the replay stream: its first indices differ;
- it yields ts_positive / ts_negative at about the replay ratio, and collated chunks are right padded;
- decision_kl on a padded chunk equals decision_kl of each row alone (padding and positions are right), and is 0 for
  the teacher against itself.
usage: decision_batch_check.py TRAIN_CONFIG STUDENT_DIR   (teacher = the config's init_model; fp32 on CPU)
"""
import random
import sys
import torch
import yaml
from swift.model import get_model_processor
from swift.template import get_template
import open_audio_llm.integrations.ms_swift.register_qwen3_asr  # noqa: F401
from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset, read_data_config
from open_audio_llm.data.catalog_sampler import CatalogBatchSampler
from ctx_kl_train import DECISION_TASKS, decision_kl, first_item

torch.set_num_threads(48)
# The sampler's signature only identifies resume cursors; a new data seed writes a new signature file into the
# metadata cache, which ACP (root) owns and the dev user cannot write. Training on ACP computes the real one.
CatalogBatchSampler._signature = lambda self, data_config, world_size: "dev-check"


def pool_in_memory(self, source, cycle, indexes):
    """CatalogBatchSampler._replay_pool without its on-disk cache (same cache, same owner problem): the same
    deterministic permutation the training run writes and loads."""
    pool = list(indexes)
    if self.shuffle:
        random.Random(f"{self.dataset.seed}:replay:{source}:{cycle}").shuffle(pool)
    return pool


CatalogBatchSampler._replay_pool = pool_in_memory
P = yaml.safe_load(open(sys.argv[1]))["parameters"]
student_dir = sys.argv[2]


def load(path):
    m, proc = get_model_processor(path, model_type="amphion_asr_1.7b", torch_dtype=torch.float32, attn_impl="sdpa",
                                  device_map="cpu")
    m.thinker.audio_tower.config._attn_implementation = "sdpa"
    return m.eval().requires_grad_(False), proc


teacher, processor = load(P["init_model"])
student = load(student_dir)[0]
template = get_template(processor, max_length=None, template_type="amphion_asr_1.7b"); template.set_mode("train")
(language,) = processor.tokenizer.encode("language", add_special_tokens=False)

rp_cfg = read_data_config(P["replay_data"]); dc_cfg = read_data_config(P["replay_data"])
dc_cfg["seed"] = int(P["decision_seed"])
firsts = []
for cfg in (rp_cfg, dc_cfg):
    s = CatalogBatchSampler(CatalogSwiftDataset(cfg, encode=template.encode, message_format="qwen3_asr"), batch_size=1,
                            rank=0, world_size=4, shuffle=True)
    s.set_epoch(0); it = iter(s); firsts.append([next(it)[0] for _ in range(20)])
same = sum(a == b for a, b in zip(*firsts))
print("first 20 indices equal at", same, "places; replay", firsts[0][:3], "decision", firsts[1][:3])
assert same < 3

ds = CatalogSwiftDataset(dc_cfg, encode=template.encode, message_format="qwen3_asr")
sampler = CatalogBatchSampler(ds, batch_size=1, rank=0, world_size=4, shuffle=True); sampler.set_epoch(0)
loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=2, collate_fn=first_item, prefetch_factor=2,
                                     generator=torch.Generator().manual_seed(int(P["decision_seed"])))
kept, seen = [], 0
for item in loader:
    seen += 1
    if int(item["catalog_task"]) in DECISION_TASKS:
        kept.append(item)
    if len(kept) == 8:
        break
print("kept", len(kept), "of", seen, "tasks", [int(x["catalog_task"]) for x in kept])
mb = P["decision_micro_batch"]
for i in range(0, len(kept), mb):
    batch = template.data_collator(kept[i:i + mb])
    am = batch["attention_mask"]
    lengths = am.sum(1)
    assert bool((am[torch.arange(am.shape[0]), lengths - 1] == 1).all()) and bool((am.cumsum(1).max(1).values == lengths).all())
    right = all(bool((am[r, :int(lengths[r])] == 1).all()) for r in range(am.shape[0]))
    print("chunk", i // mb, "lengths", lengths.tolist(), "right padded", right)
    assert right
    together = decision_kl(student, teacher, batch, language)
    alone = torch.cat([decision_kl(student, teacher, template.data_collator([x]), language) for x in kept[i:i + mb]])
    zero = decision_kl(teacher, teacher, batch, language)
    print("  together", [f"{v:.6f}" for v in together.tolist()], "alone", [f"{v:.6f}" for v in alone.tolist()],
          "teacher-teacher max", zero.abs().max().item())
    assert torch.allclose(together, alone, rtol=1e-3, atol=1e-6) and zero.abs().max().item() < 1e-6
print("ok")
