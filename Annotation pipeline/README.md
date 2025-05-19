# SolarChemQA Annotation Pipeline

The SolarChemQA annotation sources pipeline is a two-phase system designed to annotate textual evidence from solar chemistry literature for seven predefined experimental parameters. First phase is to utlize large language models to generate all possible sentences. And second phase is to invite domain expert to annotation whether the sentence actually contains the evidences.


### Annotation Sources Demo:

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

### Annotation Process

![How we Annotated~](/Annotation pipeline/img/solar_eval_pipeline.png)

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
