"""Static code compatibility check; does not import/install or train MegaTrain."""

import ast
from common import ROOT, now, write_json


def main():
    repo = ROOT / "third_party/MegaTrain"
    core = repo / "infinity/model/cpu_master.py"
    text = core.read_text()
    tree = ast.parse(text)
    methods = {
        n.name: n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    evidence = {}
    for name in [
        "_discover_model_components",
        "_process_vision",
        "_merge_vision_embeddings",
        "get_parameters",
    ]:
        n = methods[name]
        evidence[name] = dict(
            path=str(core.relative_to(ROOT)), line=n.lineno, end_line=n.end_lineno
        )
    config = ROOT / "model/TinyLLaVA-1.5B/config.json"
    import json

    c = json.loads(config.read_text()) if config.exists() else {}
    report = dict(
        time=now(),
        status="not_directly_compatible",
        runtime_smoke_test="not_run",
        checkpoint_model_type=c.get("model_type"),
        checkpoint_architectures=c.get("architectures"),
        evidence=evidence,
        blockers=[
            "Checkpoint legacy tinyllava type is not an HF llava AutoModel; explicit class registration and local loader required.",
            "Legacy attributes are model.vision_tower and model.mm_projector; Factory main uses connector. Neither matches current generic projector discovery.",
            "Legacy image sentinel -200 requires expansion into 729 image embeddings plus corresponding labels, masks and positions; generic fixed-size replacement is not equivalent.",
            "Vision encoder and projector run under torch.no_grad(): end-to-end vision/projector training is not implemented in this path.",
            "Chained boolean indexing merged[img_mask][:n_use] writes into a temporary tensor; image merge requires correctness testing and correction.",
            "requirements.txt requests Transformers >=5.0.0, conflicting with checkpoint runtime 4.37.2.",
            "GPU unavailable; CUDA extensions not built. RAM/VRAM operation has not been verified.",
        ],
        conclusion="Keep inference/evaluation in official TinyLLaVA. Do not present a TinyLLaVA MegaTrain training config as executable.",
    )
    write_json(ROOT / "docs/megatrain_compatibility.json", report)
    print(report["status"])
    for b in report["blockers"]:
        print("-", b)


if __name__ == "__main__":
    main()
