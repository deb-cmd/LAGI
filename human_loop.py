from typing import List
from agno.agent import Agent, RunResponse
from agno.models.ollama import Ollama
from pydantic import BaseModel, Field

class MultipleChoiceQuestion(BaseModel):
    question: str = Field(..., description="Clear and specific clarifying question")
    choices: List[str] = Field(..., min_items=3, max_items=4, 
                             description="3-4 relevant answer choices for the question")

class ClarificationQuestionSet(BaseModel):
    questions: List[MultipleChoiceQuestion] = Field(
        ..., 
        min_items=2, 
        max_items=3,
        description="List of multiple-choice clarifying questions to understand user intent"
    )

query_refiner = Agent(
    name="Multi-Choice Clarification Engine",
    model=Ollama(id="qwen2.5:3b"),
    description="Generates multiple clarifying questions with answer choices to pinpoint user intent",
    instructions=[
        "Identify 2-3 key aspects of the query that need clarification",
        "For each aspect, create a focused multiple-choice question",
        "Provide 3-4 relevant answer choices per question covering common possibilities",
        "Ensure questions are specific and mutually exclusive in their options",
        "Maintain neutral phrasing without assuming prior knowledge"
    ],
    response_model=ClarificationQuestionSet,
)

query_refiner.print_response(
    "What is LLM",
    stream=True,
    markdown=True,
)