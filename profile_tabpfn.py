"""Profile real TabPFN v3 inference on CUDA to verify the cost proportions:
how much of a forward pass is sample-attention vs feature-attention vs MLP,
and whether attention dominates at all.

Model: Prior-Labs/tabpfn_3 (loaded via the `tabpfn` package).

Step 1 (`inspect`): discover TabPFN-3's module structure + the shapes each
attention submodule sees, so we can attribute time correctly. Step 2
(`profile`, added next) builds the timed breakdown on top.

    uv run modal run profile_tabpfn.py::inspect
"""
import modal

# Clean image: let tabpfn pull a compatible torch (don't reuse the microbench
# image, whose torch is pinned for flash-attn's ABI). Weights download at
# runtime into the cache volume and are committed for fast subsequent runs.
CACHE = "/vol"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("tabpfn", "scikit-learn")
    # env points caches at the mounted volume. No XDG_CACHE_HOME -> nothing is
    # written into the mount during build, so the volume mounts cleanly.
    .env({
        "HF_HOME": f"{CACHE}/hf",
        "HF_HUB_CACHE": f"{CACHE}/hf/hub",
        "TABPFN_MODEL_CACHE_DIR": f"{CACHE}/tabpfn",
    })
    .add_local_python_source("profile_tabpfn")  # must be last
)

app = modal.App("tab-attn-profile", image=image)
cache_vol = modal.Volume.from_name("tabpfn-cache", create_if_missing=True)

HF_REPO = "Prior-Labs/tabpfn_3"


@app.function(gpu="H100", volumes={CACHE: cache_vol}, timeout=60 * 30)
def inspect(filename: str = ""):
    import torch
    import torch.nn as nn
    import tabpfn
    from huggingface_hub import list_repo_files, hf_hub_download
    from sklearn.datasets import make_classification
    from tabpfn import TabPFNClassifier

    import traceback

    report = {
        "tabpfn_version": getattr(tabpfn, "__version__", "?"),
        "torch_version": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "gpu": str(torch.cuda.get_device_name(0)),
    }
    try:
        return _inspect_body(report, filename, torch, nn, list_repo_files,
                             hf_hub_download, make_classification, TabPFNClassifier)
    except Exception as e:  # return errors as plain strings (never pickle torch)
        report["error"] = repr(e)
        report["traceback"] = traceback.format_exc()
        return report


def _inspect_body(report, filename, torch, nn, list_repo_files, hf_hub_download,
                  make_classification, TabPFNClassifier):
    # Pull the checkpoint straight from the HF repo (no TabPFN license wrapper),
    # then load via an absolute model_path so tabpfn skips its gated download.
    repo_files = list_repo_files(HF_REPO)
    ckpts = sorted(f for f in repo_files if f.endswith(".ckpt"))
    report["repo_ckpts"] = ckpts
    if not filename:
        clf_ckpts = [f for f in ckpts if "classifier" in f.lower()]
        multi = [f for f in clf_ckpts if "multiclass" in f.lower()]
        filename = (multi or clf_ckpts or ckpts)[0]
    report["chosen_ckpt"] = filename
    local_path = hf_hub_download(HF_REPO, filename)
    cache_vol.commit()
    report["model_path"] = local_path

    # binary dataset is safe for any classifier checkpoint
    X, y = make_classification(n_samples=256, n_features=16, n_informative=10,
                               n_classes=2, random_state=0)
    clf = TabPFNClassifier(device="cuda", n_estimators=1, model_path=local_path)
    clf.fit(X, y)

    # find the torch model living somewhere on the fitted estimator. It may be
    # nested inside lists (models_), dicts, or other objects (executor_), so we
    # traverse containers too -- but stop at the first nn.Module (don't descend
    # into its children).
    def find_modules(obj, prefix="", depth=5, seen=None):
        if seen is None:
            seen = set()
        out = []
        if depth < 0 or id(obj) in seen:
            return out
        seen.add(id(obj))
        if isinstance(obj, (str, bytes, type)):
            return out
        if isinstance(obj, dict):
            items = list(obj.items())
        elif isinstance(obj, (list, tuple)):
            items = list(enumerate(obj))
        else:
            items = list(getattr(obj, "__dict__", {}).items())
        for k, v in items:
            if isinstance(v, nn.Module):
                out.append((prefix + str(k), v))  # don't recurse into modules
            elif isinstance(v, (list, tuple, dict)) or hasattr(v, "__dict__"):
                out += find_modules(v, prefix + str(k) + ".", depth - 1, seen)
        return out

    candidates = find_modules(clf)
    candidates.sort(key=lambda kv: sum(p.numel() for p in kv[1].parameters()),
                    reverse=True)
    report["estimator_attrs"] = [a for a in vars(clf) if not a.startswith("__")][:50]
    # try to surface which checkpoint was loaded
    for attr in ("model_path", "model_name_", "checkpoint_", "model_path_"):
        v = getattr(clf, attr, None)
        if v:
            report[f"clf.{attr}"] = str(v)

    if not candidates:
        report["error"] = "no nn.Module found on fitted estimator"
        return report
    model_name, model = candidates[0]
    report["model_attr"] = model_name
    report["model_class"] = type(model).__name__
    report["n_params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)

    cfg = {}
    for src in (model, getattr(model, "config", None), clf):
        if src is None:
            continue
        for key in ("nhead", "n_heads", "num_heads", "emsize", "ninp", "d_model",
                    "nlayers", "n_layers", "num_layers", "features_per_group",
                    "max_num_features", "dropout", "attention_type"):
            v = getattr(src, key, None)
            if isinstance(v, (int, float, str)) and key not in cfg:
                cfg[key] = v
    report["config"] = cfg

    attn_mods = [(n, type(m).__name__) for n, m in model.named_modules()
                 if "attention" in type(m).__name__.lower()
                 or n.split(".")[-1].lower().startswith("attn")
                 or "attention" in n.lower()]
    report["n_attention_modules"] = len(attn_mods)
    report["attention_modules"] = attn_mods[:60]

    # hook attention modules to capture the input shape each sees during predict
    shapes = {}

    def mk_hook(name):
        def hook(mod, inputs, output):
            shp = None
            for a in inputs:
                if torch.is_tensor(a):
                    shp = tuple(a.shape)
                    break
            shapes.setdefault(name, str(shp))
        return hook

    handles = []
    attn_names = set(n for n, _ in attn_mods)
    for n, m in model.named_modules():
        if n in attn_names:
            handles.append(m.register_forward_hook(mk_hook(n)))
    _ = clf.predict(X[:64])
    torch.cuda.synchronize()
    for h in handles:
        h.remove()
    report["attention_input_shapes"] = shapes

    tree = [f"{n}: {type(m).__name__}" for n, m in model.named_modules()
            if n and n.count(".") <= 2]
    report["module_tree_top"] = tree[:90]

    cache_vol.commit()
    return report


@app.function(gpu="H100", volumes={CACHE: cache_vol}, timeout=60 * 30)
def profile(n_train: int = 4096, n_test: int = 2048, n_features: int = 64,
            n_classes: int = 2, n_estimators: int = 1, repeats: int = 3,
            filename: str = ""):
    import traceback
    import torch
    report = {
        "torch": str(torch.__version__), "cuda": str(torch.version.cuda),
        "gpu": str(torch.cuda.get_device_name(0)),
        "params": dict(n_train=n_train, n_test=n_test, n_features=n_features,
                       n_classes=n_classes, n_estimators=n_estimators,
                       repeats=repeats),
    }
    try:
        return _profile_body(report, n_train, n_test, n_features, n_classes,
                             n_estimators, repeats, filename)
    except Exception as e:
        report["error"] = repr(e)
        report["traceback"] = traceback.format_exc()
        return report


def _profile_body(report, n_train, n_test, n_features, n_classes, n_estimators,
                  repeats, filename, do_kernels=True):
    from collections import defaultdict
    import numpy as np
    import torch
    import torch.nn as nn
    from huggingface_hub import list_repo_files, hf_hub_download
    from sklearn.datasets import make_classification
    from tabpfn import TabPFNClassifier

    if not filename:
        files = list_repo_files(HF_REPO)
        cands = [f for f in files if f.endswith(".ckpt") and "classifier" in f.lower()]
        multi = [f for f in cands if "multiclass" in f.lower()]
        filename = (multi or cands)[0]
    report["ckpt"] = filename
    local_path = hf_hub_download(HF_REPO, filename)
    cache_vol.commit()

    X, y = make_classification(n_samples=n_train + n_test, n_features=n_features,
                               n_informative=max(2, n_features // 2),
                               n_classes=n_classes, random_state=0)
    Xtr, ytr, Xte = X[:n_train], y[:n_train], X[n_train:]

    clf = TabPFNClassifier(device="cuda", n_estimators=n_estimators,
                           model_path=local_path, ignore_pretraining_limits=True)
    clf.fit(Xtr, ytr)
    model = clf.models_[0]
    report["model_class"] = type(model).__name__
    report["n_icl_blocks"] = len(getattr(model, "icl_blocks", []))
    report["n_columnagg_blocks"] = len(getattr(getattr(model, "column_aggregator",
                                                       None), "blocks", []))
    report["n_feat_embed_layers"] = len(getattr(getattr(model,
                                       "feature_distribution_embedder", None),
                                       "layers", []))

    # ---- timed module categories via CUDA events ----
    # map each module instance -> category label
    cat_of = {}
    for n, m in model.named_modules():
        if n == "feature_distribution_embedder":
            cat_of[m] = "1_feature_embedder(ISAB)"
        elif n == "column_aggregator":
            cat_of[m] = "2_column_aggregator(feature-attn)"
        elif n.startswith("icl_blocks.") and n.count(".") == 1:
            cat_of[m] = "3_icl_block_total(sample-attn)"
        elif n.startswith("icl_blocks.") and n.endswith(".icl_attention"):
            cat_of[m] = "3a_icl_attention"
        elif n.startswith("icl_blocks.") and n.endswith(".mlp"):
            cat_of[m] = "3b_icl_mlp"
    full_model = model  # for total forward

    events = []  # (category, start_event, end_event)
    starts = {}

    def pre_hook(m, inp):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        starts[id(m)] = e

    def post_hook(cat):
        def f(m, inp, out):
            s = starts.get(id(m))
            if s is None:
                return
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            events.append((cat, s, e))
        return f

    handles = []
    for m, cat in cat_of.items():
        handles.append(m.register_forward_pre_hook(pre_hook))
        handles.append(m.register_forward_hook(post_hook(cat)))
    # whole-model forward timing
    handles.append(full_model.register_forward_pre_hook(pre_hook))
    handles.append(full_model.register_forward_hook(post_hook("0_model_forward")))

    # warmup
    _ = clf.predict(Xte)
    torch.cuda.synchronize()

    cat_times = defaultdict(list)
    predict_ms = []
    for _ in range(repeats):
        events.clear()
        torch.cuda.reset_peak_memory_stats()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        _ = clf.predict(Xte)
        t1.record()
        torch.cuda.synchronize()
        predict_ms.append(t0.elapsed_time(t1))
        per_cat = defaultdict(float)
        for cat, s, e in events:
            per_cat[cat] += s.elapsed_time(e)
        for k, v in per_cat.items():
            cat_times[k].append(v)
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    for h in handles:
        h.remove()

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else 0.0

    model_fwd = med(cat_times.get("0_model_forward", [0]))
    breakdown = {}
    for cat, xs in sorted(cat_times.items()):
        ms = med(xs)
        breakdown[cat] = {
            "ms_per_predict": round(ms, 3),
            "pct_of_model_fwd": round(100 * ms / model_fwd, 1) if model_fwd else None,
        }
    report["predict_ms_median"] = round(med(predict_ms), 3)
    report["model_fwd_ms_median"] = round(model_fwd, 3)
    report["peak_mem_mb"] = round(peak_mb, 1)
    report["breakdown"] = breakdown

    if not do_kernels:
        return report

    # ---- kernel-level breakdown via torch.profiler ----
    from torch.profiler import profile as torch_profile, ProfilerActivity
    with torch_profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        _ = clf.predict(Xte)
    torch.cuda.synchronize()

    def dev_time(ev):
        for attr in ("self_device_time_total", "self_cuda_time_total"):
            v = getattr(ev, attr, None)
            if v:
                return v
        return 0.0

    ops = [(ev.key, dev_time(ev)) for ev in prof.key_averages()]
    ops = [(k, t) for k, t in ops if t > 0]
    total_dev = sum(t for _, t in ops) or 1.0
    ops.sort(key=lambda x: -x[1])
    report["top_kernels"] = [
        {"op": k, "ms": round(t / 1e3, 3), "pct": round(100 * t / total_dev, 1)}
        for k, t in ops[:20]
    ]
    return report


@app.function(gpu="H100", volumes={CACHE: cache_vol}, timeout=60 * 40)
def sweep(filename: str = ""):
    import traceback
    import torch
    configs = (
        [dict(n_train=n, n_test=2048, n_features=64) for n in
         (1024, 4096, 16384, 65536)] +
        [dict(n_train=8192, n_test=2048, n_features=f) for f in (16, 64, 256)]
    )
    runs = []
    for c in configs:
        rep = {"params": dict(**c, n_classes=2, n_estimators=1, repeats=3)}
        try:
            rep = _profile_body(rep, c["n_train"], c["n_test"], c["n_features"],
                                2, 1, 3, filename, do_kernels=False)
        except Exception as e:
            rep["error"] = repr(e)
            rep["traceback"] = traceback.format_exc()
        runs.append(rep)
        torch.cuda.empty_cache()
    return {"gpu": str(torch.cuda.get_device_name(0)),
            "torch": str(torch.__version__), "runs": runs}


@app.local_entrypoint()
def main(filename: str = ""):
    import json
    rep = inspect.remote(filename)
    print(json.dumps(rep, indent=2, default=str))


@app.local_entrypoint()
def sweep_main(filename: str = ""):
    import json
    rep = sweep.remote(filename)
    print(json.dumps(rep, indent=2, default=str))


@app.local_entrypoint()
def run(n_train: int = 4096, n_test: int = 2048, n_features: int = 64,
        n_classes: int = 2, n_estimators: int = 1, repeats: int = 3,
        filename: str = ""):
    import json
    rep = profile.remote(n_train, n_test, n_features, n_classes, n_estimators,
                         repeats, filename)
    print(json.dumps(rep, indent=2, default=str))
