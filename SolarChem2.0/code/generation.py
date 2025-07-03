from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# from langchain_anthropic import ChatAnthropic
# from langchain_deepseek import ChatDeepSeek
import os
import json
from typing import List
import time
import argparse
from fuzzywuzzy import fuzz
from pydantic import BaseModel, Field

from newannotatorScript import *



choices = {
    "catalyst": "The answer should be purely chemical symbols to represent one chemical compound or element",
    "co-catalyst": "The answer should be purely chemical symbols to represent one or multiple chemical compounds or elements",
    "light source": "UV, Solar, UV-Vis, Monochromatic, Solar Simulator",
    "lamp": "Fluorescent, Mercury, Halogen, Mercury-Xenon, LED, Tungsten, Xenon, Tungsten-Halide, Solar Simulator",
    "reaction medium": "Liquid, Gas",
    "reactor type": "Slurry, Fixed-bed, Optical Fiber, Monolithic, Membrane, Fluidised-bed",
    "operation mode": "Batch, Continuous, Batch/Continuous",
}

definition = {
    "catalyst": "A substance that increases the rate of a reaction without modifying the overall standard Gibbs energy change in the reaction",
    "co-catalyst": "A substance or agent that brings about catalysis in conjunction with one or more others",
    "light source": "A light source is an optical subsystem that provides light for use in a distant area using a delivery system (e.g., fiber optics). Light sources may include one of a variety of lamps (e.g., xenon, halogen, mercury). Most light sources are operated from line power, but some may be powered from batteries. They are mostly used in endoscopic, microscopic, and other examination and/or in surgical procedures. The light source is part of the optical subsystem. In a flow-cytometer the light source directs high intensity light at particles at the interrogation point. The light source in a flow cytometer is usually a laser",
    "lamp": "A devide that generates heat, light, or any other form of radiation",
    "reaction medium": "The medium used for the reaction of the experiment execution",
    "reactor type": "The condition of the reactor devide that performs the process of a photocatalysis reaction",
    "operation mode": "The condition whether the operation is perfomed in batch-mode or continuous-mode",
}


class Answer(BaseModel):
    answer: str = Field(description="The actual value of the experiment setting")

prompt_template = "Answer the user query based on the provided evidence. \n{format_instructions}\nQuery: {query}\nExperiment Setting: {category}\nExperiment Setting Definition: {definition}\nPossible Choices: {choice}\nEvidence: {evidence}"

query = """
Here is an extracted quota from a solar chemistry paper which describes an exeperiment setting used in a solar chemistry experiment.
Can you please help me to identify the experiment setting?
"""


def run_llm_annotation(llm, parser, query, category, definition, choice, evidence):
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["query", "category", "choice", "definition", "evidence"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm
    return chain.invoke({"query": query, "category": category, "choice": choice, "definition": definition, "evidence": evidence})


major_vote_prompt_template = """
Here is a list of inferences about the {category} used in a solar chemistry paper according to a state-of-the-art LLM model. However, the correct answer is yet to be selected. To provide you with a context about the solar chemistry experiment, here are some extracted sentences from the paper, which is about the {category}. Can you choose the correct answer for me?
If no inferences or context is given, please indicate the answer as "not specific"
Choices of the answer of the {category} is: {choices}
{format_instructions}
Context: {evidences}
Inferences: {inferences}
"""

def run_major_vote(llm, parser, category, choices, inferences, evidences):
    prompt = PromptTemplate(
        template=major_vote_prompt_template,
        input_variables=["category", "inferences", "choices", "evidences"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm
    return chain.invoke({"category": category, "choices": choices, "inferences": inferences, "evidences": evidences})

def major_voting_with_range(data, llm, query, choices, definitions):
    res = {}
    parser = PydanticOutputParser(pydantic_object=Answer)
    for cate, annotation in data["annotation"].items():
        choice = choices[cate]
        defi = definitions[cate]
        res[cate] = ""
        temp_res = []
        evidences = ""
        for evis in annotation:
            evi = evis["source"]

            evidences += evi
            evidences += "\n"
            ai_msg = run_llm_annotation(llm, parser, query, cate, defi, choice, evi)
            answer = parser.parse(ai_msg.content)

            temp_res.append(answer.answer)
            time.sleep(5)
        print(temp_res)
        print("final answer is here:")
        inferences = ''.join(temp_res)
        final_ai_msg = run_major_vote(llm, parser, cate, choice, inferences, evidences)
        final_answer = parser.parse(final_ai_msg.content)
        res[cate] = final_answer.answer

        print(final_answer)
        print("finished")
    return res


def get_parser():
    parser = argparse.ArgumentParser(description="Solar Annotation Pipeline")
    parser.add_argument('--user_key', default="XXX", help="Gemini Key", type=str)
    parser.add_argument('--model_id', default = "gemini-2.5-pro", help="gemini ai model reference", type=str)
    parser.add_argument('--input_file_root', help="the location of the folder of the file being annotated", type=str)
    parser.add_argument('--evidence_file_root', help="the location of the folder for saving the annotated file", type=str)
    parser.add_argument('--save_file_dir', help="the location of the folder for saving the annotated file", type=str)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    args_dict = vars(args)
    os.environ["OPENAI_API_KEY"] = args_dict["user_key"]
    llm = ChatOpenAI(
        model=args_dict["model_id"],
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
        base_url="http://185.150.190.211:3000/v1",
    )
    
    save_file_dir = args_dict["save_file_dir"]
    for file_path in os.listdir(args_dict["input_file_root"]):
        file_index = file_path.split(".")[0].split("_")[-1]
        print(f"Processing file: {file_index}")
        save_file_loc = args_dict["save_file_dir"] + "/" + "annotated_annotation_" + file_index + ".json"
        evidence_file_loc = args_dict["evidence_file_root"] + "/" + "evidences_" + file_index + ".json"
        if os.path.exists(save_file_loc):
            pass
        else:
            root_file_path = os.path.join(args_dict["input_file_root"], file_path)
            try:
                file_data = load_json(root_file_path)
                annotation = extract(file_data, llm, extract_query, Evidences)
                time.sleep(60)
                res = major_voting_with_range(annotation, llm, query, choices, definition)
                with open(evidence_file_loc, "w") as f:
                    json.dump(annotation, f)
                with open(save_file_loc, "w") as f:
                    json.dump(res, f)
                print(res)
                time.sleep(60)
            except Exception as error:
                # handle the exception
                print("An exception occurred:", error)

main()