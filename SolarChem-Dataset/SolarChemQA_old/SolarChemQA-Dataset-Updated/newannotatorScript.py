from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# from langchain_anthropic import ChatAnthropic
# from langchain_deepseek import ChatDeepSeek
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

catalyst: The catalyst used in the experiment, often is a chemical compound
co-catalyst: The co-catalyst used in the experiment, often is a chemical compound
light source: The type of light source used in the experiment. Possible choices are UV, Solar, UV-Vis, Monochromatic, Solar Simulator
lamp: The type of lamp used in the experiment. Possible choices are Fluorescent, Mercury, 'Halogen', 'Mercury-Xenon', 'LED', 'Tungsten', 'Xenon', 'Tungsten-Halide', 'Solar Simulator'
reaction medium: The type of the medium of the reactor used in the experiment. Possible choices are 'Liquid', 'Gas'
reactor type: The type of the reactor, possible choices are 'Slurry', 'Fixed-bed', 'Optical Fiber', 'Monolithic', 'Membrane', 'Fluidised-bed'
operation mode: The mode of operation in the experiment. Possible choices are 'Batch', 'Continuous', 'Batch/Continuous'

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

class Alignment(BaseModel):
    category: str = Field(description="The category of this answer")
    alignment: str = Field(description="The alignment flag. 1 refers to align, 0 refers to not align.")


def check_alignment(extracted_source, gt_annotation, llm, alignment_prompt, parser):
    prompt = PromptTemplate(
        template="Help me analysis the user query.\n{format_instructions}\n{alignment_prompt}\nExtracted Sentence:{extracted_source}\nGround Truth {gt_annotation}",
        input_variables=["alignment_prompt", "gt_annotation", "extracted_source"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm
    return chain.invoke({"alignment_prompt": alignment_prompt, "gt_annotation": gt_annotation, "extracted_source": extracted_source})

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
                        # print(temp_answer)
                        # print(res["annotation"][temp_answer["category"]])
                        temp = {"llm generation": temp_answer["inferences"], "source": temp_answer["source"], "context":item["content"]}
                        res["annotation"][temp_answer["category"]].append(temp)
                        # print(res["annotation"][temp_answer["category"]])
                    else:
                        pass
                print("Auto-Annotation Succeed!")
            except:
                print(f"Auto-Annotation Failed :(")
    return res


def extract_pipeline(file_path, model_id, gt, extract_query, alignment_prompt, Evidences, Alignment):
    
#     llm = ChatGoogleGenerativeAI(
#         model=model_id,
#         temperature=0,
#         max_tokens=None,
#         timeout=None,
#         max_retries=2,
#         # other params...
#     )
    
    llm = ChatOpenAI(
        model=model_id,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
        base_url="http://185.150.190.211:3000/v1",
        # organization="...",
        # other params...
    )
    # llm.invoke("who are you?")
    
    # llm = ChatDeepSeek(
    #     model=model_id,
    #     temperature=0,
    #     max_tokens=None,
    #     timeout=None,
    #     max_retries=2,
    #     # other params...
    # )
    # llm = ChatAnthropic(model='claude-3-7-sonnet-latest')
    
    alignment_parser = PydanticOutputParser(pydantic_object=Alignment)
    
    ground_truth_index = file_path.split(".")[0].split("_")[-1]
    ground_truth = gt[ground_truth_index]
    file_data = load_json(file_path)
    # print(data)
    res = extract(file_data, llm, extract_query, Evidences)
    new_annos = {"paper_title": res["paper_title"], "DOI": res["DOI"], "annotator": res["human validator"], "annotations": {}}
    res_anno = res["annotation"]
    for cate, annos in res_anno.items():
        gt_annotation = ground_truth[cate]
        new_annos["annotations"][cate] = []
        for anno in annos:
            # print(anno)
            alignment = check_alignment(anno["source"], gt_annotation, llm, alignment_prompt, alignment_parser)
            # print(alignment)
            alignment_flag = dict(alignment_parser.parse(alignment.content))["alignment"]
            if alignment_flag == "1":
                new_annos["annotations"][cate].append(anno)
            else:
                pass
    return new_annos


def get_parser():
    parser = argparse.ArgumentParser(description="Solar Annotation Pipeline")
    parser.add_argument('--user_key', default="sk-VEJr8zCy6x9MlQ0yvh7rtaxSOwf3C7WDeg7CuGOahDrEKQl8", help="Gemini Key", type=str)
    parser.add_argument('--model_id', default = "gemini-2.5-flash-preview-05-20", help="gemini ai model reference", type=str)
    # parser.add_argument('--user_key', default="sk-0ed68dc9b79e45509d0923388497f418", help="Gemini Key", type=str)
    # parser.add_argument('--model_id', default = "deepseek-reasoner", help="gemini ai model reference", type=str)
    parser.add_argument('--paper_file_dir', help="the location of the folder of the file being annotated", type=str)
    parser.add_argument('--gt_file_path', help="the location for the ground truth annotation targets json file", type=str)
    parser.add_argument('--save_file_dir', help="the location of the folder for saving the annotated file", type=str)
    return parser

# (file_path, model_id, gt, extract_query, alignment_prompt, Evidences, Alignment)
def main():
    parser = get_parser()
    args = parser.parse_args()
    args_dict = vars(args)
    # os.environ["GOOGLE_API_KEY"] = args_dict["user_key"]
    os.environ["OPENAI_API_KEY"] = args_dict["user_key"]
    # os.environ["ANTHROPIC_API_KEY"] = args_dict["user_key"]
    save_file_dir = args_dict["save_file_dir"]
    gt_data = load_json(args_dict["gt_file_path"])
    
    for file_path in os.listdir(args_dict["paper_file_dir"]):
        file_index = file_path.split(".")[0].split("_")[-1]
        save_file_loc = args_dict["save_file_dir"] + "/" + "annotated_annotation_" + file_index + ".json"
        if os.path.exists(save_file_loc):
            pass
        else:
            if file_index in list(gt_data.keys()):
                model_id = args_dict["model_id"]
                root_file_path = os.path.join(args_dict["paper_file_dir"], file_path)
                print(root_file_path)
                try:
                    annotation = extract_pipeline(root_file_path, model_id, gt_data, extract_query, alignment_prompt, Evidences, Alignment)
                # print(annotation)
                # save_file_loc = args_dict["save_file_dir"] + "/" + "annotated_annotation_" + file_index + ".json"
                    with open(save_file_loc, "w") as f:
                        json.dump(annotation, f)
                    time.sleep(15)
                except Exception as error:
                    # handle the exception
                    print("An exception occurred:", error)

        
main()
        