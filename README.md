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
- paragraphs: A list of paragraphs. Each paragraph is a dictionary contains `paragraph_text` and `question_category`
    - `paragraph_text`: A paragraph from the extracted text from the academic paper
    - `question_category`: A list of question categories which are indicated in the paragraph

### Dataset Demo:

```json
[
    {
        "paper_title": "1,3,5-Triphenylbenzene Based Porous Conjugated Polymers for Highly Efficient Photoreduction of Low-Concentration CO2 in the Gas-Phase System",
        "paper_doi": "10.1002/solr.202100872",
        "paragraphs": 
        [
            {
                "paragraph_text": "XXX",
                "question_category": ["catalyst", "co-catalyst"]
            }, ...
        ]
    }, ...
]

```


