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
    build_voter_agent,
    extract_evidences,
    major_vote_answers,
)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SolarChem Generation Pipeline (agno)"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to the config YAML (default: ./config.yaml).",
    )
    return parser



def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_paper_json(path: str) -> Any:
    with open(path, "rb") as f:
        return json.load(f)


def sanitize_model_id(model_id: str) -> str:
    cleaned = re.sub(r"[\s/\\:]+", "_", model_id.strip())
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "", cleaned)
    return cleaned or "model"


def resolve_output_dirs(cfg: Dict[str, Any]) -> Tuple[Path, Path, Path]:
    paths_cfg = cfg.get("paths") or {}
    input_folder = paths_cfg.get("input_folder")
    output_folder = paths_cfg.get("output_folder")
    if not input_folder or not output_folder:
        raise ValueError(
            "config.yaml must define `paths.input_folder` and "
            "`paths.output_folder`."
        )

    model_dir_name = sanitize_model_id(get_model_id(cfg["model"]))
    model_root = Path(output_folder) / model_dir_name
    evidence_dir = model_root / "evidence"
    result_dir = model_root / "result"

    evidence_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    return Path(input_folder), evidence_dir, result_dir

def main() -> int:
    args = get_parser().parse_args()

    if not Path(args.config).is_file():
        print(f"ERROR: config file not found: {args.config}", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    model_cfg = cfg.get("model") or {}
    provider = (model_cfg.get("provider") or "").lower()
    model_id = get_model_id(model_cfg)
    retries = (cfg.get("agents") or {}).get("default_retries", 2)

    try:
        input_folder, evidence_dir, result_dir = resolve_output_dirs(cfg)
    except ValueError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2

    if not input_folder.is_dir():
        print(
            f"ERROR: input_folder is not a directory: {input_folder}",
            file=sys.stderr,
        )
        return 2

    print(f"[setup] provider={provider} model={model_id} retries={retries}")
    print(f"[setup] input_folder = {input_folder}")
    print(f"[setup] evidence_dir = {evidence_dir}")
    print(f"[setup] result_dir   = {result_dir}")

    model = build_model(model_cfg)
    extractor = build_extractor_agent(model, cfg["prompts"], retries)
    annotator = build_annotator_agent(model, cfg["prompts"], retries)
    voter = build_voter_agent(model, cfg["prompts"], retries)

    runtime = cfg["runtime"]

    for file_name in sorted(os.listdir(input_folder)):
        file_index = file_name.split(".")[0].split("_")[-1]
        save_file_loc = result_dir / f"annotated_annotation_{file_index}.json"
        evidence_file_loc = evidence_dir / f"evidences_{file_index}.json"

        if save_file_loc.exists():
            print(f"[skip] {file_name} (already processed)")
            continue

        print(f"[process] {file_name}")
        try:
            file_data = load_paper_json(str(input_folder / file_name))

            annotation = extract_evidences(file_data, extractor, cfg)
            time.sleep(runtime["inter_paper_sleep"])

            final_answers = major_vote_answers(annotation, annotator, voter, cfg)

            with open(evidence_file_loc, "w", encoding="utf-8") as f:
                json.dump(annotation, f, ensure_ascii=False, indent=2)
            with open(save_file_loc, "w", encoding="utf-8") as f:
                json.dump(final_answers, f, ensure_ascii=False, indent=2)

            print(f"[done] {file_name} -> {save_file_loc}")
            print(json.dumps(final_answers, ensure_ascii=False, indent=2))
            time.sleep(runtime["inter_paper_sleep"])
        except Exception as err:  # noqa: BLE001
            print(f"[error] {file_name}: {err}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
