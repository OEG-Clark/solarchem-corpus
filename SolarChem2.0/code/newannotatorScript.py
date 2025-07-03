from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


from langchain.agents import tool, create_react_agent, AgentExecutor
import os
import json
from typing import List
import time
import argparse
from fuzzywuzzy import fuzz
from pydantic import BaseModel, Field


extract_query = """
Please help me analysis if there are mentions in the provided text. There are 7 different types of mentions. 

catalyst: A substance that increases the rate of a reaction without modifying the overall standard Gibbs energy change in the reaction
co-catalyst: A substance or agent that brings about catalysis in conjunction with one or more others
light source: A light source is an optical subsystem that provides light for use in a distant area using a delivery system. Light sources may include one of a variety of lamps. Most light sources are operated from line power, but some may be powered from batteries. They are mostly used in endoscopic, microscopic, and other examination and/or in surgical procedures. The light source is part of the optical subsystem. In a flow-cytometer the light source directs high intensity light at particles at the interrogation point. The light source in a flow cytometer is usually a laser. Possible choices are UV, Solar, UV-Vis, Monochromatic, Solar Simulator
lamp: A devide that generates heat, light, or any other form of radiation. Possible choices are Fluorescent, Mercury, 'Halogen', 'Mercury-Xenon', 'LED', 'Tungsten', 'Xenon', 'Tungsten-Halide', 'Solar Simulator'
reaction medium: The medium used for the reaction of the experiment execution. Possible choices are 'Liquid', 'Gas'
reactor type: The condition of the reactor devide that performs the process of a photocatalysis reaction. Possible choices are 'Slurry', 'Fixed-bed', 'Optical Fiber', 'Monolithic', 'Membrane', 'Fluidised-bed'
operation mode:The condition whether the operation is perfomed in batch-mode or continuous-mode. Possible choices are 'Batch', 'Continuous', 'Batch/Continuous'

Please help me analysis if there are mentions in the provided text. If so, please indicate the answer for each mention in the provided text. And please indicate which sentence is the source of the mention as well. 

If there are any of these types are mentioned, please answer the question as type, answer and source from the context.
"""


alignment_prompt = "Can you evaluate if the extracted sentence aligned with the provided ground truth?"


class Evidence(BaseModel):
    category: str = Field(description="The category of this answer")
    inferences: str = Field(description="The actual selection or inference of the answer")
    source: str = Field(description="The original source of the answer")

class Evidences(BaseModel):
    analysis: str = Field(description="The thinking process")
    evidences: List[Evidence] = Field(description="list of answers which contains the category, answer and source context of the answer")
    
    
def load_json(file_path):
    with open(file_path, "rb") as f:
        data = json.load(f)
    return data

def run_llm_annotation(llm, parser, query, context):
    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\nContext: {context}",
        input_variables=["query", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm
    return chain.invoke({"query": query, "context": context})

def check_source(extracted_source, para):
    ratio = fuzz.partial_ratio(extracted_source, para)
    if ratio >= 100:
        return 1
    else:
        return 0

def extract(file_data, llm, query, Evidences):
    evidence_parser = PydanticOutputParser(pydantic_object=Evidences)
    res = {}
    for item in file_data:
        if item["title"] == "Doi":
            res["DOI"] = item["content"]
            doi = item["content"]
        elif item["title"] == "Article_Title":
            res["paper_title"] = item["content"]
    # res["paper_title"] = 
    res["human validator"] = "hybrid annotator"
    res["annotation"] = {"catalyst": [], "co-catalyst": [], "light source": [], "lamp": [], "reaction medium": [], "reactor type": [], "operation mode": []}
    for item in file_data:
        if item["title"] in ["Article_Title", "Abstract", "Experimental", "Results_and_Discussion", "Conclusions"]:
            answers = run_llm_annotation(llm, evidence_parser, query, item["content"])
            # print(answers)
            try:
                answers = evidence_parser.parse(answers.content)
                # print(answers)
                for answer in answers.evidences:
                    temp_answer = dict(answer)
                    ### Add checker
                    flag = check_source(temp_answer["source"], item["content"])
                    print(flag)
                    if flag == 1:
                        temp = {"llm generation": temp_answer["inferences"], "source": temp_answer["source"], "context":item["content"]}
                        res["annotation"][temp_answer["category"]].append(temp)
                    else:
                        pass
                print("Auto-Annotation Succeed!")
            except:
                print(f"Auto-Annotation Failed :(")
            time.sleep(5)
    return res


def extract_pipeline(file_path, llm, extract_query, alignment_prompt, Evidences, Alignment):
    alignment_parser = PydanticOutputParser(pydantic_object=Alignment)
    file_data = load_json(file_path)
    res = extract(file_data, llm, extract_query, Evidences)
    return res

        