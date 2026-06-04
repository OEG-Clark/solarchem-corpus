import json
import time
from typing import Any, Dict, List, Optional

from agno.agent import Agent
from fuzzywuzzy import fuzz

from rag import Ragger
from schemas import Answer, Evidences

NOT_SPECIFIC = "not specific"


def _build_agent(model, prompt_block: Dict[str, Any], output_schema, retries: int) -> Agent:
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
    return _build_agent(model, prompts["extract"], Evidences, retries)


def build_annotator_agent(model, prompts: Dict[str, Any], retries: int) -> Agent:
    return _build_agent(model, prompts["annotate"], Answer, retries)


def build_ragger(cfg: Dict[str, Any]) -> Ragger:
    """Construct a Ragger from the active config."""
    rag_cfg = cfg.get("rag") or {}
    runtime = cfg.get("runtime") or {}
    return Ragger(
        chunk_type=rag_cfg.get("chunking", "Naive"),
        rank_type=rag_cfg.get("retrieval", "Naive"),
        chunk_size=runtime.get("chunk_size", 1024),
        overlap=runtime.get("overlap", 128),
        top_k=runtime.get("top_k", 5),
        paper_sections=runtime.get("paper_sections", []),
    )


def _content_to_model(response, schema_cls):
    content = getattr(response, "content", response)
    if isinstance(content, schema_cls):
        return content
    if isinstance(content, dict):
        return schema_cls(**content)
    if isinstance(content, str):
        return schema_cls(**json.loads(content))
    raise TypeError(f"Unexpected response content type: {type(content).__name__}")


def _check_source(extracted_source: str, paragraph: str, threshold: int) -> bool:
    if not extracted_source or not paragraph:
        return False
    return fuzz.partial_ratio(extracted_source, paragraph) >= threshold


def _extract_one_category(
    extractor: Agent,
    category: str,
    context: str,
    template: str,
    runtime: Dict[str, Any],
) -> Optional[str]:
    prompt = template.format(context=context)
    try:
        response = extractor.run(prompt)
        evidences_obj = _content_to_model(response, Evidences)
    except Exception as err:
        print(f"  [extract] {category} failed: {err}")
        return None
    finally:
        time.sleep(runtime.get("inter_call_sleep", 0))

    threshold = runtime["source_match_threshold"]
    for ev in evidences_obj.evidences:
        if ev.category != category:
            continue
        if _check_source(ev.source, context, threshold):
            return ev.source
    return None


def _annotate(
    annotator: Agent,
    category: str,
    cat_meta: Dict[str, str],
    evidence_text: str,
    template: str,
    runtime: Dict[str, Any],
) -> Optional[str]:
    prompt = template.format(
        category=category,
        definition=cat_meta["definition"],
        choices=cat_meta["choices"],
        evidence=evidence_text,
    )
    try:
        resp = annotator.run(prompt)
        return _content_to_model(resp, Answer).answer
    except Exception as err:
        print(f"  [annotate] {category} failed: {err}")
        return None
    finally:
        time.sleep(runtime.get("inter_call_sleep", 0))


def generate_answers(
    file_data: List[Dict[str, Any]],
    extractor: Agent,
    annotator: Agent,
    ragger: Ragger,
    cfg: Dict[str, Any],
    paper_index: Optional[str] = None,
) -> Dict[str, str]:
    runtime = cfg["runtime"]
    categories_meta = cfg["categories"]
    extract_template = cfg["prompts"]["extract"]["prompt_template"]
    annotate_template = cfg["prompts"]["annotate"]["prompt_template"]
    queries = cfg.get("queries") or {}

    ragger.build_for_paper(file_data, paper_index=paper_index)

    answers: Dict[str, str] = {}

    if not ragger.input_data.strip():
        print("  [retrieve] no usable section text in this paper")
        for category in categories_meta:
            answers[category] = NOT_SPECIFIC
        return answers

    for category, cat_meta in categories_meta.items():
        query = queries.get(category)
        if not query:
            print(f"  [retrieve] no query configured for '{category}', skipping")
            answers[category] = NOT_SPECIFIC
            continue

        try:
            retrieved_texts = ragger.retrieve(query)
        except Exception as err:
            print(f"  [retrieve] {category} failed: {err}")
            answers[category] = NOT_SPECIFIC
            continue

        if not retrieved_texts:
            print(f"  [retrieve] {category}: no chunks retrieved")
            answers[category] = NOT_SPECIFIC
            continue

        context = "\n\n".join(retrieved_texts)
        evidence_source = _extract_one_category(
            extractor, category, context, extract_template, runtime
        )
        if evidence_source is None:
            answers[category] = NOT_SPECIFIC
            print(f"  [done] {category}: {NOT_SPECIFIC} (no verifiable evidence)")
            continue

        ans = _annotate(
            annotator, category, cat_meta, evidence_source, annotate_template, runtime
        )
        answers[category] = ans if ans is not None else NOT_SPECIFIC
        print(f"  [done] {category}: {answers[category]!r}")

    return answers
