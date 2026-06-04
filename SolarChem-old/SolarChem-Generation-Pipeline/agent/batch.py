import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, List

from generate import load_config, run_pipeline
from model_factory import get_model_id


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def normalise_run_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    rag_overlay: Dict[str, Any] = {}
    model_overlay: Dict[str, Any] = {}

    for key, value in entry.items():
        if key in ("chunking", "retrieval"):
            rag_overlay[key] = value
        elif key == "provider":
            model_overlay["provider"] = value
        elif key in ("ollama", "openrouter", "deepseek", "max_tokens", "temperature"):
            model_overlay[key] = value
        elif key == "rag":
            rag_overlay = deep_merge(rag_overlay, value or {})
        elif key == "model":
            model_overlay = deep_merge(model_overlay, value or {})
        else:
            out[key] = value

    if rag_overlay:
        out["rag"] = rag_overlay
    if model_overlay:
        out["model"] = model_overlay
    return out


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the SolarChem pipeline over a list of overlays on config.yaml."
    )
    p.add_argument(
        "--batch",
        required=True,
        help="Path to the batch YAML (e.g. batch_rag.yaml or batch_eval.yaml).",
    )
    p.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to the base config YAML (default: ./config.yaml).",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running subsequent batch entries after a failure.",
    )
    return p


def main() -> int:
    args = get_parser().parse_args()

    if not Path(args.config).is_file():
        print(f"ERROR: base config not found: {args.config}", file=sys.stderr)
        return 2
    if not Path(args.batch).is_file():
        print(f"ERROR: batch file not found: {args.batch}", file=sys.stderr)
        return 2

    base_cfg = load_config(args.config)
    batch_cfg = load_config(args.batch)

    runs: List[Dict[str, Any]] = batch_cfg.get("runs") or []
    if not runs:
        print(
            "ERROR: batch file must contain a non-empty `runs:` list.",
            file=sys.stderr,
        )
        return 2

    batch_wide_override = {k: v for k, v in batch_cfg.items() if k != "runs"}
    batch_wide_override = normalise_run_entry(batch_wide_override)

    merged_runs: List[Dict[str, Any]] = []
    for entry in runs:
        run_override = normalise_run_entry(entry)
        merged = deep_merge(base_cfg, batch_wide_override)
        merged = deep_merge(merged, run_override)
        merged_runs.append(merged)

    model_ids = set()
    rag_tags = set()
    for m in merged_runs:
        try:
            model_ids.add(get_model_id(m["model"]))
        except Exception:
            pass
        rag = m.get("rag") or {}
        rag_tags.add((rag.get("chunking"), rag.get("retrieval")))

    skip_model_dir = len(model_ids) <= 1
    skip_rag_dir = len(rag_tags) <= 1

    for m in merged_runs:
        paths = m.setdefault("paths", {})
        paths["_skip_model_dir"] = skip_model_dir
        paths["_skip_rag_dir"] = skip_rag_dir

    print(f"[batch] {len(runs)} run(s) queued")
    print(f"[batch] base config: {args.config}")
    print(f"[batch] batch  file: {args.batch}")
    if skip_model_dir and not skip_rag_dir:
        print(f"[batch] output layout: <output_folder>/<chunking>-<retrieval>/")
    elif skip_rag_dir and not skip_model_dir:
        print(f"[batch] output layout: <output_folder>/<model>/")
    elif skip_model_dir and skip_rag_dir:
        print(f"[batch] output layout: <output_folder>/")
    else:
        print(f"[batch] output layout: <output_folder>/<model>/<chunking>-<retrieval>/")

    failures: List[str] = []
    for i, merged in enumerate(merged_runs, start=1):
        try:
            model_id = get_model_id(merged["model"])
        except Exception as err:
            print(f"[batch] run {i}: invalid model config: {err}", file=sys.stderr)
            failures.append(f"run {i}: bad config")
            if args.continue_on_error:
                continue
            return 2

        rag = merged.get("rag") or {}
        tag = f"{model_id} | {rag.get('chunking')}-{rag.get('retrieval')}"
        print(f"\n[batch] ===== run {i}/{len(merged_runs)}: {tag} =====")
        try:
            rc = run_pipeline(merged)
            if rc != 0:
                failures.append(f"run {i} ({tag}): exit code {rc}")
                if not args.continue_on_error:
                    return rc
        except Exception as err:
            print(f"[batch] run {i} ({tag}) crashed: {err}", file=sys.stderr)
            failures.append(f"run {i} ({tag}): {err}")
            if not args.continue_on_error:
                return 1

    print(f"\n[batch] all runs complete")
    if failures:
        print(f"[batch] {len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
