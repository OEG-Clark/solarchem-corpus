# SolarChem QA-RAG Benchmark & Dataset

This repository contains the benchmark and dataset for [SolarQA](https://github.com/oeg-upm/solar-qa) application. The SolarChem QA-RAG Benchmark is designed specifically for factual question answering benchmark based on solar chemistry academic papers. We intent to build an open-source benchmark & dataset for testing the performance of QA system when it comes to narrow-domain and in-depth questions. 

## Dataset

The dataset was extracted from more than 1000 academic papers about solar chemistry experiments. And there are 7 types of questions for each experiment. 

### Questions

The questions from this dataset cover seven categories/aspects from solar chemistry experiments, which are indicated below. And the questions can be devided into two types: open-ended and close-ended, due to if there is a range of answers to be choose from.

| Question Type |  Category  |  Range of Choices  |
| --------      | -------    | -------    |
| open-ended    | catalyst        |  None  |
| open-ended    | co-catalyst     |  None  |
| close-ended   | light source    |  'UV', 'Solar', 'UV-Vis', 'Monochromatic', 'Solar Simulator', 'Do not Know'  |
| close-ended   | lamp            |  'Fluorescent', 'Mercury', 'Halogen', 'Mercury-Xenon', 'LED', 'Tungsten', 'Xenon', 'Tungsten-Halide', 'Solar Simulator', 'Do not Know'  |
| close-ended   | reaction medium |  'Liquid', 'Gas', 'Do not Know'  |
| close-ended   | reactor type    |  'Slurry', 'Fixed-bed', 'Optical Fiber', 'Monolithic', 'Membrane', 'Fluidised-bed', 'Do not Know'  |
| close-ended   | operation mode  |  'Batch', 'Continuous', 'Batch/Continuous', 'Do not Know'  |


### Dataset Structure

The dataset consist of:
- Paper Information: Indication of paper, which contains `paper_title` and `paper_doi`
    - `paper_title`: The title of the solar chemistry paper
    - `paper_doi`: The DOI of the paper
- `paragraphs`: A list of paragraphs. Each paragraph is a dictionary contains `paragraph_text` and `annotations`
    - `paragraph_text`: A paragraph from the extracted text from the academic paper
    - `annotations`: A list of objects that contains `annotator`, `category`, `answer` and `context`.
        - `annotator`: Annotator reference, noted that `hybrid annotation` refers to the annotation based on large language model and a human.
        - `category`: The question category of this object
        - `answer`: The answer of the question based on the `paragraph_text`
        - `context`: The original context in the `paragraph_text` which mentioned or indicated the `answer`

### Dataset Demo:

```json

{
    "paper_title": "1,3,5-Triphenylbenzene Based Porous Conjugated Polymers for Highly Efficient Photoreduction of Low-Concentration CO2 in the Gas-Phase System",
    "paper_doi": "10.1002/solr.202100872",
    "paragraphs": [
        {
            "paragraph_text": "Developing near-infrared responsive (NIR) photocatalysts is very important for the development of solardriven photocatalytic systems.Metal sulfide semiconductors have been extensively used as visible-light responsive photocatalysts for photocatalytic applications owing to their high chemical variety, narrow bandgap and suitable redox potentials, particularly the benchmark ZnIn 2 S 4 .However, their potential as NIR-responsive photocatalysts is yet to be reported.Herein, for the first time demonstrated that upconversion nanoparticles can be delicately coupled with hierarchical ZnIn 2 S 4 nanorods (UCNPs/ZIS) to assemble a NIR-responsive composite photocatalyst, and as such composite is verified by ultraviolet-visible diffuse reflectance spectra and upconversion luminescence spectra.As a result, remarkable photocatalytic CO and CH 4 production rates of 1500 and 220 nmol g A1 h A1 , respectively, were detected for the UCNPs/ZIS composite under NIR-light irradiation (k !800 nm), which is rarely reported in the literature.The remarkable photocatalytic activity of the UCNPs/ZIS composite can be understood not only because the heterojunction between UCNPs and ZIS can promote the charge separation efficiency, but also the intimate interaction of UCNPs with hierarchical ZIS nanorods can enhance the energy transfer.This finding may open a new avenue to develop more NIR-responsive photocatalysts for various solar energy conversion applications.",
            "annotations": [
                {
                    "annotator": "hybrid annotation",
                    "category":"catalyst",
                    "answer":"UCNPs/ZIS composite",
                    "context": "Herein, for the first time demonstrated that upconversion nanoparticles can be delicately coupled with hierarchical ZnIn₂S₄ nanorods (UCNPs/ZIS) to assemble a NIR-responsive composite photocatalyst"
                },
                {
                    "annotator": "hybrid annotation",
                    "category":"light source",
                    "answer":"Monochromatic",
                    "context": "NIR-light irradiation (k !800 nm)"
                },...
            ]
        }, ...
    ]
}

```


## Annotation

### Annotation Process

![How we Annotated~](img/solar_eval_pipeline.png "How we Annotated~")

1. Use the LLM to apply auto annotation
2. Pass the answer to the human evaluator
3. Human evaluate check the llm annotated answers as the final annotation


### Annotation Script

- The script is based on [Gemini 2.0 Flash](https://deepmind.google/technologies/gemini/flash/) model, which is a state of the art generative model with outstanding reasoning capacity. 
- The script is based on the [Solar-QA-Pipeline](https://github.com/oeg-upm/solar-qa/tree/main/CLI) extraction from solar chemistry academic papers, which is build upon [grobid](https://github.com/kermitt2/grobid)
- The script is formatted as jupyter notebook, before get started, please make sure the `path` for files and folders are correct, and then start the annotation!


```
langchain
langchain-google-genai
langchain-community
langchain-core
pydantic
```

Poetry Support: 
1. Install [Poetry](https://python-poetry.org/)
2. `cd code`
3. `poetry install`

### Script Arguement
`python annotator.py --user_key XXX --model_id gemini-2.0-flash --file_dir ../paper_1025.json --save_file_dir test.json`

Run with Poetry:

`poetry run python annotator.py --user_key XXX --model_id gemini-2.0-flash --file_dir ../paper_1025.json --save_file_dir test.json`

Here is a table that describe the parameters to run the SolarAnnotator

| Parameter | Definition | DataType | Reference/Example |
| -------- | ------- | ------- | ------- |
| user_key  | the key of Gemini AI platform | String | [Gemini](https://gemini.google.com/) |
| model_id | the reference of gemini model | String | [model_id](https://ai.google.dev/gemini-api/docs/models/gemini) |
| file_dir | path for input data as a json format | String | ../paper_1025.json |
| save_file_dir | path for where to save the result | String | ..test.json |