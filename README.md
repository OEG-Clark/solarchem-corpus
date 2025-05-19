# SolarChem QA-RAG Benchmark & Dataset

This repository contains the benchmark and dataset for [SolarQA](https://github.com/oeg-upm/solar-qa) application. The SolarChem QA-RAG Benchmark is designed specifically for factual question answering benchmark based on solar chemistry academic papers. We intent to build an open-source benchmark & dataset for testing the performance of QA system when it comes to narrow-domain and in-depth questions. 

## SolarChemQA Dataset

SolarChemQA provides a novel question answering dataset curated from solar chemistry literature, designed to rigorously assess the capabilities of LLM-driven QA systems in processing domain-specific scientific content. The dataset includes 574 domain expert annotations across 82 solar chemistry research papers, structured around seven critical experimental parameters. Together with 289 annotated source sentences with positive and negative examples across 21 papers.

### Sub-Datasets

- Query Sub-Dataset: Standardized queries for the seven experimental parameter categories
- Paper Extraction Sub-Dataset: Structured content extracted from research papers
- Annotation Sub-Dataset: 574 expert-validated annotations and 289 supporting textual sentences

## SolarChemQA Evaluation

This repository also provides comprehensive benchmark tasks and evaluation methodologies for the SolarChemQA dataset. The benchmark is designed to evaluate LLM-driven QA systems across three critical dimensions in solar chemistry literature.

### Task 1: Information Retrieval Evaluation
Assesses the effectiveness of different retrieval strategies in accessing relevant information from solar chemistry papers.

```json
Evaluation Target:
- Chunking methodologies: Naive, Recursive, and Semantic Segmentation
- Retrieval strategies: Naive, Hybrid, and Reranking approaches
- Performance at identifying relevant experimental contexts
```
```json
Metric: Normalized Discounted Cumulative Gain (NDCG)
```

Run Evaluation: 
Arguments:
- input_file_dir: Path of raw extracted paper contents folder
- gt_file_dir: Path of the source annotation folder
- res_file_dir: Path for saving the evaluated result folder
- chunk_size: The maximum number of characters or tokens allowed in a single chunk
-overlap: Overlap between chunks ensures that information at the boundaries is not lost or contextually isolated.
```python
python Eval_IR.py --input_file_dir ".../SolarChemQA-Dataset/extracted_paper" --gt_file_dir ".../SolarChemQA-Dataset/domain_expert_annotated_sources/" --res_file_dir ".../Task-1-information-retrieval/result/" --chunk_size "1028" --overlap "128"
```

### Task 2: RAG Method Evaluation
Measures the effectiveness of different Retrieval-Augmented Generation strategies for providing accurate answers.

```json
Evaluation Target:
- Nine combinations of chunking-retrieval methods
- Answer quality across all seven experimental parameter categories
- Semantic fidelity and lexical precision of generated answers
```
```json
Metrics:
- Semantic similarity (cosine similarity between embeddings)
- Lexical matching (partial_ratio algorithm)
```
Run Evaluation: 
Arguments:
- input_file_dir: Path of raw extracted paper contents folder
- res_file_dir: Path for saving the evaluated result folder
- chunk_type: Naive, Recursive, Semantic.
- rank_type: Naive, Rerank, Hybrid.
- chunk_size: The maximum number of characters or tokens allowed in a single chunk
-overlap: Overlap between chunks ensures that information at the boundaries is not lost or contextually isolated.
```python
python Eval_RAG.py --input_file_dir ".../SolarChemQA-Dataset/extracted_paper" --res_file_dir ".../Task-1-information-retrieval/result/" --chunk_type "Naive" --rank_type "Naive" --chunk_size "1028" --overlap "128"
```

### Task 3: LLM Performance Evaluation
Compares the capabilities of various LLMs in understanding and answering questions about solar chemistry experiments.

```json
Evaluation Target:
- Performance of API-based models: gemini-2.0-flash, gemini-2.5-flash, deepseek-v3, deepseek-r1, qwen3-plus, qwen2.5-max
- Performance of locally-run models: deepseek-r1-32b, qwen-30b, gemma3-27b
- Ability to capture experimental settings from scientific literature
```
```json
Metrics:
- Semantic similarity (cosine similarity between embeddings)
- Lexical matching (partial_ratio algorithm)
```
Run Evaluation: 
Arguments:
- input_file_dir: Path of raw extracted paper contents folder
- res_file_dir: Path for saving the evaluated result folder
- chunk_type: Naive, Recursive, Semantic.
- rank_type: Naive, Rerank, Hybrid.
- chunk_size: The maximum number of characters or tokens allowed in a single chunk
-overlap: Overlap between chunks ensures that information at the boundaries is not lost or contextually isolated.
```python
python Eval_LLM.py --input_file_dir ".../SolarChemQA-Dataset/extracted_paper" --res_file_dir ".../Task-1-information-retrieval/result/" --chunk_type "Naive" --rank_type "Naive" --chunk_size "1028" --overlap "128"
```

## SolarChemQA Benchmark

Our benchmark results are available in the results/ directory, including:
- Comparative analysis of retrieval strategies
- Performance metrics for RAG methodologies
- LLM performance rankings
  
### Task 1:

| Question Category | Chunking Method |      |     | Retrieval Method |           |           |
|-------------------|----------------|------|-----|------------------|-----------|-----------|
|                   | Naive          | Recursive | Semantic         | Naive     | Hybrid    | Reranking |
| Catalyst          | 0.3913         | 0.3141    | **0.4056**       | 0.3994    | **0.4250**| 0.2216    |
| Co-Catalyst       | 0.2605         | 0.2424    | **0.3141**       | 0.2456    | **0.3168**| 0.1714    |
| Light Source      | 0.3142         | 0.2884    | **0.3784**       | 0.3356    | **0.3856**| 0.3273    |
| Lamp              | 0.3416         | 0.2997    | **0.4026**       | 0.3385    | **0.4736**| 0.4381    |
| Reactor Type      | 0.2758         | 0.2597    | **0.3110**       | 0.2801    | **0.3640**| 0.3045    |
| Reaction Medium   | 0.2643         | 0.2612    | **0.3164**       | 0.2689    | **0.3621**| 0.2316    |
| Operation Mode    | 0.2421         | 0.2798    | **0.3056**       | 0.2957    | **0.3432**| 0.3750    |
| Average           | 0.2985         | 0.2779    | **0.3477**       | 0.3091    | **0.3815**| 0.2836    |


### Task 2:

| Chunking-Retrieval | Metric | Catalyst | Co-Catalyst | Light Source | Lamp | Reactor Type | Reaction Medium | Operation Mode | Average |
|--------------------|--------|----------|-------------|--------------|------|--------------|-----------------|----------------|---------|
| Naive-Naive        | cos_sim | 0.6830 | 0.6218 | 0.7059 | 0.9338 | 0.5661 | 0.8339 | 0.8258 | 0.7386 |
|                    | ratio   | 0.3902 | 0.3415 | 0.4024 | 0.8415 | 0.0854 | 0.5732 | 0.5610 | 0.4565 |
| Naive-Hybrid       | cos_sim | 0.7016 | 0.6221 | 0.7280 | 0.9296 | 0.5639 | 0.8233 | 0.7570 | 0.7322 |
|                    | ratio   | 0.4756 | 0.3049 | 0.4268 | 0.8049 | 0.0732 | 0.5610 | 0.3780 | 0.4321 |
| Naive-Rerank       | cos_sim | 0.6593 | 0.6428 | 0.7222 | 0.7990 | 0.5704 | 0.7786 | 0.7253 | 0.6997 |
|                    | ratio   | 0.4268 | 0.3780 | 0.4390 | 0.5122 | 0.0488 | 0.4512 | 0.2927 | 0.3641 |
| Recursive-Naive    | cos_sim | 0.6649 | 0.6051 | 0.7013 | 0.9419 | 0.5839 | 0.7889 | 0.8189 | 0.7293 |
|                    | ratio   | 0.3780 | 0.3293 | 0.3537 | 0.8659 | 0.0976 | 0.5122 | 0.5366 | 0.4390 |
| Recursive-Hybrid   | cos_sim | 0.6834 | 0.6191 | 0.7299 | 0.9307 | 0.5707 | 0.8281 | 0.7079 | 0.7243 |
|                    | ratio   | 0.5000 | 0.3049 | 0.4634 | 0.8293 | 0.0854 | 0.5732 | 0.2439 | 0.4286 |
| Recursive-Rerank   | cos_sim | 0.6807 | 0.6170 | 0.7361 | 0.7967 | 0.5769 | 0.7647 | 0.6762 | 0.6926 |
|                    | ratio   | 0.4146 | 0.3293 | 0.4878 | 0.5122 | 0.0610 | 0.3902 | 0.1585 | 0.3362 |
| Semantic-Naive     | cos_sim | 0.6868 | 0.6215 | 0.7188 | 0.8769 | 0.5546 | 0.8622 | 0.8208 | 0.7345 |
|                    | ratio   | 0.4390 | 0.3293 | 0.4268 | 0.6829 | 0.0732 | 0.6220 | 0.5610 | 0.4477 |
| Semantic-Hybrid    | cos_sim | 0.6901 | 0.6308 | 0.6708 | 0.8998 | 0.5639 | 0.8786 | 0.7632 | 0.7282 |
|                    | ratio   | 0.5366 | 0.4512 | 0.3902 | 0.7561 | 0.0854 | 0.6585 | 0.4146 | 0.4703 |
| Semantic-Rerank    | cos_sim | 0.6640 | 0.6449 | 0.7370 | 0.8083 | 0.5607 | 0.8400 | 0.7495 | 0.7149 |
|                    | ratio   | 0.4268 | 0.3780 | 0.5122 | 0.5244 | 0.0366 | 0.5854 | 0.3415 | 0.4007 |

Task 3:

![leaderboard~](/solarchem-corpus/img/llm_performance_solarchemqa "leaderboard~")
