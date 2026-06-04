from typing import List
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    category: str = Field(description="Category this evidence belongs to")
    inferences: str = Field(
        description="The actual selection or inference of the answer"
    )
    source: str = Field(
        description="The exact verbatim sentence(s) from the section that "
                    "support the inference"
    )


class Evidences(BaseModel):
    analysis: str = Field(description="Brief description of the reasoning")
    evidences: List[Evidence] = Field(
        default_factory=list,
        description="List of evidences extracted from the section",
    )


class Answer(BaseModel):
    answer: str = Field(description="The actual value of the experiment setting")
