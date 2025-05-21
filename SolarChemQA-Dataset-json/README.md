---
license: apache-2.0
task_categories:
- question-answering
language:
- en
tags:
- SolarChemistry
- LLM
pretty_name: SolarChemQA
size_categories:
- n<1K
---

# SolarChemQA

## Dataset Description

SolarChemQA is a specialized question-answering (QA) dataset curated from 82 solar chemistry research articles, designed to evaluate the performance of Large Language Model (LLM)-driven QA systems in processing domain-specific scientific literature. The dataset focuses on seven experimental parameter categories commonly found in solar chemistry experiments, providing a standardized framework to assess retrieval, integration, and reasoning capabilities of QA systems.

The dataset is structured into three interconnected sub-datasets: Query, Paper Extraction, and Annotation, as detailed below. It includes 574 domain expert annotations and 289 validated annotation sources, offering ground-truth reliability for evaluating LLM-driven QA systems.

### Query:
Query sub-dataset contains seven queries corresponding to experimental parameters in solar chemistry literature: Catalyst, Co-Catalyst, Light Source, Lamp, Reactor Medium, Reactor Condition, Operation Mode

| Experiment Setting  | Definition |
| ------------- | ------------- |
| Catalyst  | A substance that increases the rate of a reaction without modifying the overall standard Gibbs energy change in the reaction  |
| Co-Catalyst  | A substance or agent that brings about catalysis in conjunction with one or more others  |
| Light Source  | A light source is an optical subsystem that provides light for use in a distant area using a delivery system (e.g., fiber optics).  |
| Lamp  | A devide that generates heat, light, or any other form of radiation  |
| Reactor Medium  | The medium used for the reaction of the experiment execution  |
| Reactor Condition  | The condition of the reactor devide that performs the process of a photocatalysis reaction  |
| Operation Mode  | The condition whether the operation is perfomed in batch-mode or continuous-mode  |


### Paper Extraction:
This sub-dataset contains the raw extracted context from 82 solar chemistry papers. Each paper at least have one experiment, which covers 7 experiment settings. The sections we have extracted: Paper Title, Paper Doi, Abstract, Experimental, Results and Discussion, Conclusion


Data Format: A list of sections with their titles and extracted content
- Title: Section Title
- Content: Extracted text from the section.


### Annotations:
This sub-dataset contains the domain expert annotations according to the aforementioned 7 experiment settings (Annotation Targets), and sentences can be used for validating the annotation targets from the papers (Annotation Sources)

#### Annotation Targets:

Annotation Targets is regarded as the actual setting for all seven experimental setting mentioned above. With the annotated experiment settings, SolarChemQA dataset provides 574 annotation targets. In the scope of SolarChemQA, a set of vocabulary for each experiment settings is provided.


| Experiment Setting  | Range of Selection |
| ------------- | ------------- |
| Catalyst  | Open-Ended, no range is given  |
| Co-Catalyst  | Open-Ended, no range is given  |
| Light Source  | 'UV', 'Solar', 'UV-Vis', 'Monochromatic', 'Solar Simulator', 'Do not Know' |
| Lamp  | 'Fluorescent', 'Mercury', 'Halogen', 'Mercury-Xenon', 'LED', 'Tungsten', 'Xenon', 'Tungsten-Halide', 'Solar Simulator', 'Do not Know' |
| Reactor Medium  | 'Liquid', 'Gas', 'Do not Know' |
| Reactor Condition  | 'Slurry', 'Fixed-bed', 'Optical Fiber', 'Monolithic', 'Membrane', 'Fluidised-bed', 'Do not Know' |
| Operation Mode  | 'Batch', 'Continuous', 'Batch/Continuous', 'Do not Know' |

Data Format: 
- Category: The experimental setting category.
- Annotation: The annotated value for the experiment setting.

#### Annotation Sources:

Annotation Sources contains 289 validated sentences from 21 papers, providing evidence for annotation targets.

Data Format:
- Category: The experimental setting category.
- Source: Sentences with evidence supporting the annotation.
- Context: Textual content from the paper’s section.
- Vote: Validator’s judgment on the source’s accuracy.
