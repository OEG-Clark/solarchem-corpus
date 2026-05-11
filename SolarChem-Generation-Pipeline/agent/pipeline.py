import json
import time
from typing import Any, Dict, List

from agno.agent import Agent
from fuzzywuzzy import fuzz

from schemas import Answer, Evidence, Evidences


# -----------------------------------------------------------------------------
# Agent factories
# -----------------------------------------------------------------------------
def _build_agent(model, prompt_block: Dict[str, Any], output_schema, retries: int) -> Agent:
    """Common agno Agent constructor — pulls name / role / instructions from
    the YAML prompt block."""
    return Agent(
        name=prompt_block.get("name", "Agent"),
        role=prompt_block.get("role"),
        model=model,
        instructions=prompt_block["instructions"],
        output_schema=output_schema,
        use_json_mode=True,
        markdown=False,
        retries=retries,
    )


def build_extractor_agent(model, prompts: Dict[str, Any], retries: int) -> Agent:
    """Agent that reads a paper section and produces an `Evidences` object."""
    return _build_agent(model, prompts["extract"], Evidences, retries)


def build_annotator_agent(model, prompts: Dict[str, Any], retries: int) -> Agent:
    """Agent that turns one piece of evidence into one candidate `Answer`."""
    return _build_agent(model, prompts["annotate"], Answer, retries)


def build_voter_agent(model, prompts: Dict[str, Any], retries: int) -> Agent:
    """Agent that aggregates candidate answers into a final `Answer`."""
    return _build_agent(model, prompts["major_vote"], Answer, retries)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _content_to_model(response, schema_cls):
    """Coerce an agno RunResponse / RunOutput into a pydantic model.

    Agno populates ``response.content`` with an instance of ``output_schema``
    when the model honours the schema, but we fall back to JSON parsing for
    providers that occasionally return a string.
    """
    content = getattr(response, "content", response)
    if isinstance(content, schema_cls):
        return content
    if isinstance(content, dict):
        return schema_cls(**content)
    if isinstance(content, str):
        return schema_cls(**json.loads(content))
    raise TypeError(
        f"Unexpected response content type: {type(content).__name__}"
    )


def _check_source(extracted_source: str, paragraph: str, threshold: int) -> bool:
    """Verify that an `extracted_source` really appears inside `paragraph`."""
    if not extracted_source or not paragraph:
        return False
    return fuzz.partial_ratio(extracted_source, paragraph) >= threshold


# -----------------------------------------------------------------------------
# Stage 1 — extract evidences from a paper
# -----------------------------------------------------------------------------
def extract_evidences(
    file_data: List[Dict[str, Any]],
    extractor: Agent,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Walk through one paper's JSON and return the raw annotation skeleton."""
    runtime = cfg["runtime"]
    categories = list(cfg["categories"].keys())

    res: Dict[str, Any] = {
        "DOI": "",
        "paper_title": "",
        "human validator": runtime["human_validator_label"],
        "annotation": {cat: [] for cat in categories},
    }

    # Pull DOI / title out of the paper file.
    for item in file_data:
        if item.get("title") == runtime["doi_field"]:
            res["DOI"] = item.get("content", "")
        elif item.get("title") == runtime["title_field"]:
            res["paper_title"] = item.get("content", "")

    user_template = cfg["prompts"]["extract"]["prompt_template"]
    threshold = runtime["source_match_threshold"]

    for item in file_data:
        if item.get("title") not in runtime["paper_sections"]:
            continue
        section_text = item.get("content", "")
        if not section_text:
            continue

        prompt = user_template.format(context=section_text)
        try:
            response = extractor.run(prompt)
            evidences_obj = _content_to_model(response, Evidences)
        except Exception as err:  # noqa: BLE001
            print(f"  [extract] section '{item.get('title')}' failed: {err}")
            time.sleep(runtime["inter_call_sleep"])
            continue

        for ev in evidences_obj.evidences:
            if ev.category not in res["annotation"]:
                # Skip categories the model invented that aren't in our schema.
                continue
            if not _check_source(ev.source, section_text, threshold):
                continue
            res["annotation"][ev.category].append({
                "llm generation": ev.inferences,
                "source": ev.source,
                "context": section_text,
            })
        print(f"  [extract] section '{item.get('title')}' done "
              f"(+{len(evidences_obj.evidences)} candidate evidences)")
        time.sleep(runtime["inter_call_sleep"])

    return res


# -----------------------------------------------------------------------------
# Stage 2 — majority-vote final answer per category
# -----------------------------------------------------------------------------
def major_vote_answers(
    annotation: Dict[str, Any],
    annotator: Agent,
    voter: Agent,
    cfg: Dict[str, Any],
) -> Dict[str, str]:
    """Produce one final answer per category from the harvested evidences."""
    runtime = cfg["runtime"]
    categories = cfg["categories"]
    annotate_template = cfg["prompts"]["annotate"]["prompt_template"]
    vote_template = cfg["prompts"]["major_vote"]["prompt_template"]

    final: Dict[str, str] = {}

    for category, meta in categories.items():
        evidences_list = annotation["annotation"].get(category, [])
        if not evidences_list:
            final[category] = "not specific"
            print(f"  [vote] {category}: no evidence -> not specific")
            continue

        # Per-evidence single-shot annotations.
        candidates: List[str] = []
        all_evidence_texts: List[str] = []
        for ev in evidences_list:
            evidence_text = ev["source"]
            all_evidence_texts.append(evidence_text)
            prompt = annotate_template.format(
                category=category,
                definition=meta["definition"],
                choices=meta["choices"],
                evidence=evidence_text,
            )
            try:
                resp = annotator.run(prompt)
                candidates.append(_content_to_model(resp, Answer).answer)
            except Exception as err:  # noqa: BLE001
                print(f"  [annotate] {category} failed on one evidence: {err}")
            time.sleep(runtime["inter_call_sleep"])

        if not candidates:
            final[category] = "not specific"
            print(f"  [vote] {category}: all annotations failed -> not specific")
            continue

        # Aggregate candidates into a final answer.
        prompt = vote_template.format(
            category=category,
            choices=meta["choices"],
            evidences="\n".join(all_evidence_texts),
            inferences="\n".join(candidates),
        )
        try:
            resp = voter.run(prompt)
            final[category] = _content_to_model(resp, Answer).answer
        except Exception as err:  # noqa: BLE001
            print(f"  [vote] {category} aggregation failed: {err}")
            # Fall back to first candidate so we don't lose the work.
            final[category] = candidates[0]

        print(f"  [vote] {category}: candidates={candidates} -> {final[category]!r}")
        time.sleep(runtime["inter_call_sleep"])

    return final
