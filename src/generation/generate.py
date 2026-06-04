import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from model_factory import build_model, get_model_id
from pipeline import (
    build_annotator_agent,
    build_extractor_agent,
    build_ragger,
    generate_answers,
)


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SolarChem generation pipeline (agno)")
    p.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to the config YAML (default: ./config.yaml).",
    )
    return p


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_paper_json(path: str) -> Any:
    with open(path, "rb") as f:
        return json.load(f)


def sanitize_id(value: str) -> str:
    cleaned = re.sub(r"[\s/\\:]+", "_", value.strip())
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "", cleaned)
    return cleaned or "value"


def _rag_tag(cfg: Dict[str, Any]) -> str:
    rag = cfg.get("rag") or {}
    return f"{sanitize_id(rag.get('chunking', 'Naive'))}-{sanitize_id(rag.get('retrieval', 'Naive'))}"


def resolve_output_dir(cfg: Dict[str, Any]) -> Tuple[Path, Path]:
    paths_cfg = cfg.get("paths") or {}
    input_folder = paths_cfg.get("input_folder")
    output_folder = paths_cfg.get("output_folder")
    if not input_folder or not output_folder:
        raise ValueError(
            "config.yaml must define `paths.input_folder` and `paths.output_folder`."
        )

    skip_model_dir = bool(paths_cfg.get("_skip_model_dir", False))
    skip_rag_dir = bool(paths_cfg.get("_skip_rag_dir", False))

    result_dir = Path(output_folder)
    if not skip_model_dir:
        result_dir = result_dir / sanitize_id(get_model_id(cfg["model"]))
    if not skip_rag_dir:
        result_dir = result_dir / _rag_tag(cfg)

    result_dir.mkdir(parents=True, exist_ok=True)
    return Path(input_folder), result_dir


def run_pipeline(cfg: Dict[str, Any]) -> int:
    """Execute the pipeline once for the (model, rag) selection in `cfg`."""
    model_cfg = cfg.get("model") or {}
    provider = (model_cfg.get("provider") or "").lower()
    model_id = get_model_id(model_cfg)
    retries = (cfg.get("agents") or {}).get("default_retries", 2)

    try:
        input_folder, result_dir = resolve_output_dir(cfg)
    except ValueError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2

    if not input_folder.is_dir():
        print(f"ERROR: input_folder is not a directory: {input_folder}", file=sys.stderr)
        return 2

    rag_tag = _rag_tag(cfg)
    print(f"[setup] provider={provider} model={model_id} retries={retries}")
    print(f"[setup] rag={rag_tag}")
    print(f"[setup] input_folder = {input_folder}")
    print(f"[setup] result_dir   = {result_dir}")

    model = build_model(model_cfg)
    extractor = build_extractor_agent(model, cfg["prompts"], retries)
    annotator = build_annotator_agent(model, cfg["prompts"], retries)
    ragger = build_ragger(cfg)

    runtime = cfg["runtime"]

    for file_name in sorted(os.listdir(input_folder)):
        file_index = file_name.split(".")[0].split("_")[-1]
        save_file_loc = result_dir / f"annotated_annotation_{file_index}.json"

        if save_file_loc.exists():
            print(f"[skip] {file_name} (already processed)")
            continue

        print(f"[process] {file_name}")
        try:
            file_data = load_paper_json(str(input_folder / file_name))

            result = generate_answers(
                file_data, extractor, annotator, ragger, cfg, paper_index=file_index
            )

            with open(save_file_loc, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"[done] {file_name} -> {save_file_loc}")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            time.sleep(runtime.get("inter_paper_sleep", 0))
        except Exception as err:
            print(f"[error] {file_name}: {err}")

    return 0


def main() -> int:
    args = get_parser().parse_args()
    if not Path(args.config).is_file():
        print(f"ERROR: config file not found: {args.config}", file=sys.stderr)
        return 2
    cfg = load_config(args.config)
    return run_pipeline(cfg)


if __name__ == "__main__":
    raise SystemExit(main())