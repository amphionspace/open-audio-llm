"""Per-sample gradients of the replay KL at a drifted student (CPU; README "后续 2" 分支: why the decision term did not
lower the decision KL - gradient size and direction). For replay TS samples, the gradient over the trained parameters of
  dec = KL(teacher || student) at the presence-decision token, and
  tok = the sample's token-mean KL (what the 11-26 / "CE 只训正文" arms anchored),
and per task: mean norm, mean pairwise cosine between samples (exact, from the sum of unit vectors), cosine of the mean
gradient with the drift theta_student - theta_teacher (positive = descent pulls back toward the teacher), and the
cosine between the per-sample dec and tok gradients.
usage: decision_grad_coherence.py REPLAY_DATA_YAML STUDENT_DIR N_PER_TASK   (teacher = 11.17; frozen as the training)
"""
import collections, math, sys, time
import torch
import torch.nn.functional as F
from swift.model import get_model_processor
from swift.template import get_template
import open_audio_llm.integrations.ms_swift.register_qwen3_asr  # noqa: F401
from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset, read_data_config
from open_audio_llm.data.catalog_sampler import CatalogBatchSampler

TEACHER = "/workspace/data/ae-state/model-cache/entries/model-afea989da22b77adc0ce9365"
TASKS = ("asr", "ts_positive", "ts_negative", "sot", "asr_hotwords", "asr_no_kl")  # as ctx_kl_train.py
DROP = ("catalog_task", "twin_enroll_features", "ts_activity")
FROZEN = ("thinker.model.embed_tokens", "thinker.audio_tower.conv2d1", "thinker.audio_tower.conv2d2",
          "thinker.audio_tower.conv2d3", "thinker.audio_tower.conv_out")  # configs/train-real-deckl.yaml
WANT = ("ts_negative", "ts_positive")
torch.set_num_threads(48)
data_yaml, student_dir, n_per = sys.argv[1], sys.argv[2], int(sys.argv[3])


def load(path):
    m, proc = get_model_processor(path, model_type="amphion_asr_1.7b", torch_dtype=torch.float32, attn_impl="sdpa",
                                  device_map="cpu")
    m.thinker.audio_tower.config._attn_implementation = "sdpa"
    return m, proc


student, processor = load(student_dir)
teacher = load(TEACHER)[0].eval().requires_grad_(False)
student.eval()  # no dropout: per-sample gradients of the same function
params, drift = [], []
t_params = dict(teacher.named_parameters())
for name, p in student.named_parameters():
    if name.startswith(FROZEN):
        p.requires_grad_(False); continue
    p.requires_grad_(True); params.append(p); drift.append((p.detach() - t_params[name].detach()).flatten())
drift = torch.cat(drift); print("trained params", drift.numel(), "drift norm", drift.norm().item(), flush=True)
template = get_template(processor, max_length=None, template_type="amphion_asr_1.7b"); template.set_mode("train")
(language,) = processor.tokenizer.encode("language", add_special_tokens=False)
ds = CatalogSwiftDataset(read_data_config(data_yaml), encode=template.encode, message_format="qwen3_asr")
sampler = CatalogBatchSampler(ds, batch_size=1, rank=0, world_size=4, shuffle=True); sampler.set_epoch(0)

acc = {(t, k): {"unit": torch.zeros_like(drift), "sum": torch.zeros_like(drift), "norms": [], "kl": [], "cos_drift": []}
       for t in WANT for k in ("dec", "tok")}
cross = collections.defaultdict(list)
count = collections.Counter(); t0 = time.time()
for b in sampler:
    if all(count[t] >= n_per for t in WANT):
        break
    batch = template.data_collator([ds[j] for j in b])
    task = TASKS[int(batch["catalog_task"][0])]
    if task not in WANT or count[task] >= n_per:
        continue
    batch["input_features"] = batch["input_features"].float()
    labels = batch["labels"][:, 1:]; mask = labels != -100
    inputs = {k: v for k, v in batch.items() if k not in DROP and k != "labels"}
    with torch.no_grad():
        t_logits = teacher(**inputs, use_cache=False).logits[:, :-1][mask].float()
    s_logits = student(**inputs, use_cache=False).logits[:, :-1][mask].float()
    tok = F.kl_div(s_logits.log_softmax(-1), t_logits.log_softmax(-1), log_target=True, reduction="none").sum(-1)
    order = mask.cumsum(1)[mask]
    assert bool((labels[mask][order == 1] == language).all())
    losses = {"dec": tok[order == 2].mean(), "tok": tok.mean()}
    grads = {}
    for k, loss in losses.items():
        g = torch.autograd.grad(loss, params, retain_graph=(k == "dec"), allow_unused=True)
        g = torch.cat([(x if x is not None else torch.zeros_like(p)).flatten() for x, p in zip(g, params)])
        grads[k] = g; a = acc[(task, k)]; n = g.norm()
        a["norms"].append(n.item()); a["kl"].append(loss.item())
        if n > 0:
            a["unit"] += g / n; a["cos_drift"].append(F.cosine_similarity(g, drift, dim=0).item())
        a["sum"] += g
    cross[task].append(F.cosine_similarity(grads["dec"], grads["tok"], dim=0).item())
    count[task] += 1
    print(f"{task} {count[task]} dec {losses['dec'].item():.5f} tok {losses['tok'].item():.5f} "
          f"|g_dec| {acc[(task, 'dec')]['norms'][-1]:.4f} |g_tok| {acc[(task, 'tok')]['norms'][-1]:.4f} "
          f"{time.time() - t0:.0f}s", flush=True)

print("== summary (student", student_dir, ")")
for (task, k), a in acc.items():
    N = len([x for x in a["norms"] if x > 0]); U = a["unit"].norm().item() ** 2
    pair = (U - N) / (N * (N - 1)) if N > 1 else float("nan")
    mean_cos_drift = F.cosine_similarity(a["sum"], drift, dim=0).item()
    kl = sorted(a["kl"])
    print(f"{task} {k}: n {len(a['norms'])}, KL mean {sum(kl) / len(kl):.5f} median {kl[len(kl) // 2]:.5f} max {kl[-1]:.4f}; "
          f"|g| mean {sum(a['norms']) / len(a['norms']):.4f} max {max(a['norms']):.4f}; mean pairwise cos {pair:.4f}; "
          f"cos(mean g, drift) {mean_cos_drift:+.4f}; per-sample cos(g, drift) mean {sum(a['cos_drift']) / max(1, len(a['cos_drift'])):+.4f}")
for task, c in cross.items():
    print(f"{task}: per-sample cos(g_dec, g_tok) mean {sum(c) / len(c):.4f}")
