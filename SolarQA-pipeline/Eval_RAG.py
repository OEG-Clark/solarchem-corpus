from langchain_core.documents import Document
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM
from langchain_anthropic import ChatAnthropic

from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker


from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors.flashrank_rerank import FlashrankRerank
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers import BM25Retriever, EnsembleRetriever


from sentence_transformers import util
from pydantic import BaseModel, Field


import numpy as np
import argparse
import json
import math
import time
import os

os.environ["GOOGLE_API_KEY"] = "xxx"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


query_index = {
    "catalyst": "what is the catalyst used in the experiment?",
    "co_catalyst": "what is the co_catalyst used in the experiment?",
    "light source": "what is the light source used in the experiment?",
    "lamp": "what is the lamp used in the experiment?",
    "reaction medium": "what is the reaction medium used in the experiment?",
    "reactor type": "what is the reactor type for the experiment?",
    "operation mode": "what is the operation mode for the experiment?"
}

selection_index = {
    "catalyst": "no range of selection is provided",
    "co_catalyst": "no range of selection is provided",
    "light source": "Possible choices are: UV, Solar, UV-Vis, Monochromatic, Solar Simulator",
    "lamp": "Possible choices are: Fluorescent, Mercury, Halogen, Mercury-Xenon, LED, Tungsten, Xenon, Tungsten-Halide, Solar Simulator",
    "reaction medium": "Possible choices are: Liquid, Gas",
    "reactor type": "Possible choices are: Slurry, Fixed-bed, Optical Fiber, Monolithic, Membrane, Fluidised-bed",
    "operation mode": "Possible choices are: Batch, Continuous, Batch/Continuous"
}

prompt = """
Please help me answer the query based on the provided selections by analyzing the provided text.
If you can not analyzing the text to provide an answer, please stating the answer as "None"
\n{format_instructions}
Query: {query}
Selection: {selection}
Context: {Context}
"""

class Answer(BaseModel):
    category: str = Field(description="The category of this answer")
    answer: str = Field(description="The actual selection or inference of the answer")

parser = PydanticOutputParser(pydantic_object=Answer)

prompt = PromptTemplate(
    template=prompt,
    input_variables=["query", "selection", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

class Ragger:
    def __init__(self, input_path, chunk_type, rank_type, chunk_size, overlap):
        self.paper_index = input_path.split("_")[-1].split(".")[0]
        self.input_data = self.get_text(input_path)
        # self.ground_truth_path = ground_truth_folder + f"annotation_{self.paper_index}.json"
        self.chunk_type = chunk_type
        self.rank_type = rank_type
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embeddings = OllamaEmbeddings(model="avr/sfr-embedding-mistral:latest")
    
    def get_text(self, input_path):
        title_list = ["Article_Title", "Abstract", "Experimental", "Results_and_discussion", "Conclusions"]
        f = open(input_path, "rb")
        data = json.load(f)
        text = ""
        for item in data:
            if item["title"] in title_list:
                text += f"""{item["title"]}:"""
                text += "\n"
                text += item["content"]
                text += "\n"
        return text
    
    def get_chunks(self, chunk_type):
        if chunk_type == "Naive":
            text_splitter = CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
                separator='', 
                strip_whitespace=False
            )
        elif chunk_type == "Recursive":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
                length_function=len,
                is_separator_regex=False,
            )
        # elif chunk_type == "Semantic":
        else:
            number_of_chunk = int(math.ceil(len(self.input_data) / self.chunk_size))
            min_chunk_size = int(math.ceil(self.chunk_size / 2))
            text_splitter = SemanticChunker(self.embeddings, number_of_chunks=number_of_chunk, min_chunk_size=min_chunk_size)
        
        texts = text_splitter.split_text(self.input_data)

        documents = [Document(page_content=x) for x in texts]
        return documents

    def ranking(self, chunks, rank_type, query):
        # self.chunks = self.get_chunks()
        retriever = FAISS.from_documents(chunks, self.embeddings).as_retriever(search_kwargs={"k": len(chunks)})
        if rank_type == "Naive":
            retri_docs = retriever.invoke(query)
            retri_text = [x.page_content for x in retri_docs]
        elif rank_type == "Rerank":
            # ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
            # compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=3)
            compressor = FlashrankRerank(model="rank-T5-flan", top_n=len(chunks))
            # compressor = CohereRerank(model="rerank-english-v3.0")
            compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
            compressed_docs = compression_retriever.invoke(query)
            retri_text = [x.page_content for x in compressed_docs]
        else:
            keyword_retriever = BM25Retriever.from_documents(chunks)
            ensemble_retriever = EnsembleRetriever(retrievers=[retriever,keyword_retriever],weights=[0.5, 0.5])
            retri_docs = ensemble_retriever.invoke(query)
            retri_text = [x.page_content for x in retri_docs]
        retri_index = {k: retri_text[k] for k in range(len(retri_text))}
        return retri_index

    def judege()

    
    def rag(self, k, query_type, llm):
        query = query_index[query_type]
        selection = selection_index[query_type]
        chunks = self.get_chunks(self.chunk_type)
        retri_index = self.ranking(chunks, self.rank_type, query)
        # print(retri_index)
        
        if k > len(retri_index):
            k = len(retri_index)
        
        rag_context = ""
        for i in range(k):
            rag_context += retri_index[i]
            rag_context += "\n"
        
        chain = prompt | llm
        answer = chain.invoke({"query": query, "selection": selection, "Context": rag_context})
        return answer

# index_dir = "/home/jovyan/SolarChemEval_Neurips/IR_Eval/result/"
# input_dir = "/home/jovyan/SolarChemEval_Neurips/extracted_papers/"
# gt_dir = "/home/jovyan/Solar/CLI/new_paper/annotation_with_gt/"
    
def get_parser():
    parser = argparse.ArgumentParser(description="Solar Annotation Pipeline")
    parser.add_argument('--input_file_dir', default="/home/jovyan/SolarChemEval/data/extracted_paper/", help="Path of raw extracted paper contents folder", type=str)
    # parser.add_argument('--gt_file_dir', default = "/home/jovyan/Solar/CLI/new_paper/annotation_with_gt/", help="Path of the source annotation", type=str)
    parser.add_argument('--res_file_dir', default = "/home/jovyan/SolarChemEval/rag_result/", help="Path for saving the evaluated result per paper", type=str)
    parser.add_argument('--chunk_type', default = "Naive", help="Chunk type: Naive, Recursive, Semantic.", type=str)
    parser.add_argument('--rank_type', default = "Naive",help="Rank Type: Naive, Rerank, Hybrid", type=str)
    parser.add_argument('--chunk_size', default = 1024, help="the location for saving the annotated file", type=int)
    parser.add_argument('--overlap', default = 128, help="the location for saving the annotated file", type=int)
    return parser


def main():
    key_list = list(query_index.keys())
    para_parser = get_parser()
    args = para_parser.parse_args()
    args_dict = vars(args)
    input_file_list = os.listdir(args_dict["input_file_dir"])
    # gt_file_list = os.listdir(args_dict["gt_file_dir"])
    chunk_rank_type = args_dict["chunk_type"] + "_" + args_dict["rank_type"]
    save_file_path = args_dict["res_file_dir"] + chunk_rank_type + "/"
    try:
        os.mkdir(save_file_path)
    except:
        pass
    chunk_size = args_dict["chunk_size"]
    overlap = args_dict["overlap"]
    # input_file_list = os.listdir()
    for input_file in input_file_list:
        if input_file[-4:] == "json":
            index = input_file.split("_")[-1].split(".")[0]
            input_file_path = args_dict["input_file_dir"] + input_file
            rag = Ragger(input_file_path, args_dict["chunk_type"], args_dict["rank_type"], chunk_size, overlap)
            temp_dict = {}
            answer_file = save_file_path + f"result_{index}.json"
            if os.path.exists(answer_file):
                pass
            else:
                for key in key_list:
                    ans = rag.rag(k=5, query_type=key, llm=llm)
                    final_answer = parser.parse(ans.content).answer
                    temp_dict[key] = final_answer
                print(index)
                print(temp_dict)
                try:
                    with open(answer_file, "w") as f:
                        json.dump(temp_dict, f)
                except:
                    print("An exception occurred")
                time.sleep(30)
        else:
            pass
        # break

        
main()
                
        
    
