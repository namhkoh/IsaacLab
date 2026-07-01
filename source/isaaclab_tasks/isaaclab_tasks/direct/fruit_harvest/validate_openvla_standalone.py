"""Standalone OpenVLA validation — no Isaac Sim needed. Uses saved camera frames."""
import os, sys, glob, re
import numpy as np
import torch
from PIL import Image

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(_SCRIPT_DIR, "camera_frames")
# Fallback to hardcoded path if __file__ doesn't resolve
if not os.path.isdir(FRAMES_DIR):
    FRAMES_DIR = r"D:\research\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\fruit_harvest\camera_frames"
MODEL = "openvla/openvla-7b"
DEVICE = "cuda:0"


def _patch_transformers_compat():
    """Inline copy of OpenVLAWrapper._patch_transformers_compat() to avoid isaaclab imports."""
    import transformers
    tv = getattr(transformers, "__version__", "0")
    if not tv.startswith("5"):
        return
    hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "modules", "transformers_modules")
    # Patch processing_prismatic.py
    pattern = os.path.join(hf_cache, "openvla", "**", "processing_prismatic.py")
    for fpath in glob.glob(pattern, recursive=True):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                src = f.read()
            old_import = "from transformers.tokenization_utils import"
            new_import = "from transformers.tokenization_utils_base import"
            if old_import in src:
                src = src.replace(old_import, new_import)
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(src)
                print(f"[VLA] Patched {fpath}")
        except OSError:
            pass
    # Patch modeling_prismatic.py
    pattern_model = os.path.join(hf_cache, "openvla", "**", "modeling_prismatic.py")
    for fpath in glob.glob(pattern_model, recursive=True):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                src = f.read()
            changed = False
            if 'if timm.__version__ not in {' in src:
                src = re.sub(
                    r'if timm\.__version__ not in \{.*?\}:\s*raise NotImplementedError\([^)]*\)',
                    'pass  # timm version check relaxed for 1.x compat',
                    src, flags=re.DOTALL)
                changed = True
            if "def tie_weights(self) -> None:" in src:
                src = src.replace("def tie_weights(self) -> None:", "def tie_weights(self, **kwargs) -> None:")
                changed = True
            if changed:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(src)
                print(f"[VLA] Patched {fpath}")
        except OSError:
            pass
    # setattr fallback
    try:
        import transformers.tokenization_utils as _tok_utils
        from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
        for name, obj in [("PaddingStrategy", PaddingStrategy), ("PreTokenizedInput", PreTokenizedInput),
                          ("TextInput", TextInput), ("TruncationStrategy", TruncationStrategy)]:
            if not hasattr(_tok_utils, name):
                setattr(_tok_utils, name, obj)
    except Exception:
        pass


def patch_and_load():
    """Load OpenVLA with transformers 5.x compat patches."""
    _patch_transformers_compat()

    from transformers import AutoConfig, AutoProcessor
    try:
        from transformers import AutoModelForVision2Seq as AutoVLAModel
    except ImportError:
        from transformers import AutoModelForImageTextToText as AutoVLAModel

    print(f"[VLA] Loading {MODEL}...")
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    if hasattr(config, "auto_map"):
        if "AutoModelForVision2Seq" in config.auto_map and "AutoModelForImageTextToText" not in config.auto_map:
            config.auto_map["AutoModelForImageTextToText"] = config.auto_map["AutoModelForVision2Seq"]
    vla = AutoVLAModel.from_pretrained(
        MODEL, config=config, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True, attn_implementation="eager",
    ).to(DEVICE)
    print("[VLA] Model loaded on GPU.\n")
    return vla, processor


def predict(vla, processor, img, prompt, unnorm_key="bridge_orig", do_sample=False, **kwargs):
    """Run single inference, return raw action numpy array."""
    inputs = processor(prompt, img).to(DEVICE, dtype=torch.bfloat16)
    return vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=do_sample, **kwargs)


def predict_with_tokens(vla, processor, img, prompt, unnorm_key="bridge_orig"):
    """Run inference and return token-level details."""
    inputs = processor(prompt, img).to(DEVICE, dtype=torch.bfloat16)
    gen = vla.generate(
        inputs["input_ids"], pixel_values=inputs.get("pixel_values"),
        max_new_tokens=7, do_sample=False, output_scores=True, return_dict_in_generate=True,
    )
    tok_ids = gen.sequences[0, -7:].cpu().numpy()
    vs = vla.vocab_size
    disc = np.clip(vs - tok_ids - 1, 0, vla.bin_centers.shape[0] - 1)
    norm = vla.bin_centers[disc]

    entropies, top5s = [], []
    for s in gen.scores:
        p = torch.softmax(s[0].float(), dim=-1)
        ap = p[-256:]
        entropies.append(-(ap * (ap + 1e-10).log()).sum().item())
        t = torch.topk(ap, 5)
        top5s.append((t.indices.cpu().numpy(), t.values.cpu().numpy()))

    raw = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return tok_ids, disc, norm, raw, entropies, top5s


def load_sim_images():
    """Load saved sim frames."""
    imgs = {}
    for p in sorted(glob.glob(os.path.join(FRAMES_DIR, "vla_input_step*.png")))[:3]:
        imgs[os.path.basename(p)] = Image.open(p).convert("RGB")
    for name in ["scene_rgb_latest.png", "wrist_rgb_latest.png"]:
        p = os.path.join(FRAMES_DIR, name)
        if os.path.exists(p):
            imgs[name] = Image.open(p).convert("RGB")
    return imgs


# ========== EXPERIMENTS ==========

def exp1(vla, processor, sim_imgs):
    print("=" * 70)
    print("EXP 1: VISUAL GROUNDING — Does model respond to different images?")
    print("=" * 70)
    prompt = "In: What action should the robot take to pick up the red ball?\nOut:"
    test = {
        "solid_red":   Image.new("RGB", (224, 224), (255, 0, 0)),
        "solid_green": Image.new("RGB", (224, 224), (0, 255, 0)),
        "solid_grey":  Image.new("RGB", (224, 224), (128, 128, 128)),
        "noise":       Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)),
    }
    for k, v in sim_imgs.items():
        test[k] = v.resize((224, 224))

    results = {}
    for name, img in test.items():
        a = predict(vla, processor, img, prompt)
        results[name] = a
        print(f"  {name:30s} | action=[{', '.join(f'{v:.5f}' for v in a)}]")

    arr = np.stack(list(results.values()))
    std = arr.std(axis=0)
    print(f"\n  Per-dim STD: [{', '.join(f'{s:.6f}' for s in std)}]")
    ok = std.max() > 0.001
    print(f"  VERDICT: {'RESPONSIVE' if ok else 'NOT RESPONSIVE — model ignores images'}\n")
    return ok


def _get_test_img(sim_imgs):
    """Get a test image from sim frames or generate a synthetic one."""
    if sim_imgs:
        return list(sim_imgs.values())[0].resize((224, 224))
    return Image.fromarray(np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8))


def exp2(vla, processor, sim_imgs):
    print("=" * 70)
    print("EXP 2: TOKEN INSPECTION — What bins? How confident?")
    print("=" * 70)
    prompt = "In: What action should the robot take to pick up the red ball?\nOut:"
    img = _get_test_img(sim_imgs)

    tok_ids, disc, norm, raw, ent, top5s = predict_with_tokens(vla, processor, img, prompt)
    dims = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]
    center = len(vla.bin_centers) // 2

    print(f"  Token IDs:   {tok_ids}")
    print(f"  Bin indices: {disc}")
    print(f"  Normalized:  [{', '.join(f'{v:.4f}' for v in norm)}]")
    print(f"  Unnorm:      [{', '.join(f'{v:.5f}' for v in raw)}]")
    print()
    for i, d in enumerate(dims):
        bi, bp = top5s[i]
        c = "CENTER!" if abs(disc[i] - center) < 5 else ""
        print(f"    {d:4s}: bin={disc[i]:3d} {c:8s} entropy={ent[i]:.2f}  "
              f"top5_bins={bi}  top5_probs=[{', '.join(f'{p:.3f}' for p in bp)}]")

    avg = np.mean(ent)
    print(f"\n  Avg entropy: {avg:.2f}")
    if avg > 4.0:
        print("  VERDICT: HIGH ENTROPY — very uncertain\n")
    elif avg > 2.0:
        print("  VERDICT: MODERATE ENTROPY — some preference\n")
    else:
        print("  VERDICT: LOW ENTROPY — confident predictions\n")


def exp3(vla, processor, sim_imgs):
    print("=" * 70)
    print("EXP 3: PROMPT SWEEP — Simple vs complex instructions")
    print("=" * 70)
    img = _get_test_img(sim_imgs)

    prompts = {
        "pick_ball":      "In: What action should the robot take to pick up the red ball?\nOut:",
        "grasp_object":   "In: What action should the robot take to grasp the red object?\nOut:",
        "move_forward":   "In: What action should the robot take to move forward?\nOut:",
        "move_down":      "In: What action should the robot take to move down?\nOut:",
        "close_gripper":  "In: What action should the robot take to close the gripper?\nOut:",
        "open_gripper":   "In: What action should the robot take to open the gripper?\nOut:",
        "complex_full":   "In: What action should the robot take to pick the red sphere fruit from the plant and carefully place it into the purple basket?\nOut:",
    }
    for name, p in prompts.items():
        a = predict(vla, processor, img, p)
        print(f"  {name:16s} | action=[{', '.join(f'{v:.5f}' for v in a)}]")
    print()


def exp4(vla, processor, sim_imgs):
    print("=" * 70)
    print("EXP 4: UNNORM KEY COMPARISON — bridge_orig vs Franka datasets")
    print("=" * 70)
    img = _get_test_img(sim_imgs)
    prompt = "In: What action should the robot take to pick up the red ball?\nOut:"

    keys = [
        "bridge_orig",
        "nyu_franka_play_dataset_converted_externally_to_rlds",
        "furniture_bench_dataset_converted_externally_to_rlds",
        "fractal20220817_data",
        "taco_play",
    ]
    for key in keys:
        try:
            a = predict(vla, processor, img, prompt, unnorm_key=key)
            mag = np.abs(a[:6]).mean()
            print(f"  {key:55s} | pos_mag={mag:.5f} | action=[{', '.join(f'{v:.5f}' for v in a)}]")
        except Exception as e:
            print(f"  {key:55s} | ERROR: {e}")
    print()


def exp5(vla, processor, sim_imgs):
    print("=" * 70)
    print("EXP 5: SAMPLING — Greedy vs temperature")
    print("=" * 70)
    img = _get_test_img(sim_imgs)
    prompt = "In: What action should the robot take to pick up the red ball?\nOut:"

    a = predict(vla, processor, img, prompt)
    print(f"  greedy       | action=[{', '.join(f'{v:.5f}' for v in a)}]")

    for temp in [0.1, 0.3, 0.5, 0.7, 1.0]:
        actions = []
        for _ in range(5):
            a = predict(vla, processor, img, prompt, do_sample=True, temperature=temp)
            actions.append(a)
        arr = np.stack(actions)
        m, s = arr.mean(0), arr.std(0)
        print(f"  T={temp:.1f} (5 samp) | mean=[{', '.join(f'{v:.5f}' for v in m)}] | "
              f"std=[{', '.join(f'{v:.5f}' for v in s)}]")
    print()


def exp6(vla, processor, sim_imgs):
    print("=" * 70)
    print("EXP 6: CAMERA SOURCE — Wrist vs Scene (BridgeData uses 3rd person)")
    print("=" * 70)
    prompt = "In: What action should the robot take to pick up the red ball?\nOut:"

    for name in ["wrist_rgb_latest.png", "scene_rgb_latest.png"]:
        if name not in sim_imgs:
            print(f"  {name}: NOT AVAILABLE")
            continue
        img = sim_imgs[name].resize((224, 224))
        arr = np.array(img)
        tok, disc, norm, raw, ent, _ = predict_with_tokens(vla, processor, img, prompt)
        print(f"  {name:25s} | img_mean={arr.mean():.1f} img_std={arr.std():.1f}")
        print(f"    bins={disc} | action=[{', '.join(f'{v:.5f}' for v in raw)}]")
        print(f"    entropies=[{', '.join(f'{e:.2f}' for e in ent)}]")
    print()


if __name__ == "__main__":
    print(f"[DEBUG] FRAMES_DIR = {FRAMES_DIR}")
    print(f"[DEBUG] FRAMES_DIR exists = {os.path.isdir(FRAMES_DIR)}")
    if os.path.isdir(FRAMES_DIR):
        print(f"[DEBUG] Files in FRAMES_DIR: {os.listdir(FRAMES_DIR)[:5]}...")

    vla, processor = patch_and_load()
    sim_imgs = load_sim_images()
    print(f"Loaded {len(sim_imgs)} saved frames: {list(sim_imgs.keys())}\n")

    exp1(vla, processor, sim_imgs)
    exp2(vla, processor, sim_imgs)
    exp3(vla, processor, sim_imgs)
    exp4(vla, processor, sim_imgs)
    exp5(vla, processor, sim_imgs)
    exp6(vla, processor, sim_imgs)

    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
