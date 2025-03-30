from agno.agent import Agent, RunResponse
from agno.models.ollama import Ollama
from typing import Dict, Any
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.team.team import Team
from typing import List
from rich.pretty import pprint
from pydantic import BaseModel, Field
from agno.utils.pprint import pprint_run_response

manager_llm = "qwen2.5:3b"
reasoner_llm = "qwen2.5:3b"
coder_llm = "qwen2.5:3b"
searcher_llm = "qwen2.5:3b"
summarizer_llm = "gemma3:4b"

query = "What are LLMs?"

class SubQuery(BaseModel):
    sub_query: str = Field(..., 
        description="Specific, self-contained question to investigate part of the original query")
    category: int = Field(...,
        description="0=Search (factual/web), 1=Code (technical/implementation), 2=Reason (analysis/comparison)")

class SubQueries(BaseModel):
    sub_queries: List[SubQuery] = Field(...,
        description="Complete set of subqueries needed to fully address the original question")

manager = Agent(
    name="Query Decomposition Engine",
    model=Ollama(id=manager_llm),
    description="Specialized AI for breaking down complex questions into executable sub-components",
    instructions=[
        "Analyze the original query to identify all implicit and explicit information needs",
        "Decompose into smallest practical subqueries that can be independently investigated",
        "Categorize each subquery by type: Search (0), Code (1), or Reason (2)",
        "Ensure coverage of all aspects needed for comprehensive answer construction",
        "Maintain logical progression between subqueries where dependencies exist",
        "Avoid overlap or duplication between subqueries",
        "Prioritize clarity and specificity in subquery formulation",
        "Include comparative analysis subqueries when evaluating options/methods"
    ],
    add_datetime_to_instructions=True,
    response_model=SubQueries
)


response: RunResponse = manager.run(query)
pprint_run_response(response, markdown=True)

searcher = Agent(
    name="Research Analyst",
    role="Information Retrieval & Web Research Specialist",
    model=Ollama(id=searcher_llm),
    description=(
        "Expert in comprehensive web research and source evaluation. "
        "Specializes in finding credible, up-to-date information from diverse sources. "
        "Skilled in using advanced search operators and filtering results by date/relevance."
    ),
    tools=[DuckDuckGoTools()],
    instructions=[
        "Prioritize .gov/.edu domains and authoritative sources",
        "Verify information across multiple credible sources",
        "Highlight potential biases in sources when relevant",
        "Include publication dates and context timeliness",
        "Extract key statistics, quotes, and factual data"
    ],
    add_datetime_to_instructions=True,
)

coder = Agent(
    name="Software Engineer",
    role="Technical Implementation Specialist",
    model=Ollama(id=coder_llm),
    description=(
        "Full-stack development expert focusing on clean, maintainable code. "
        "Specializes in API integration, algorithm optimization, and debugging. "
        "Creates production-ready solutions with security best practices."
    ),
    instructions=[
        "Follow PEP8/standard style guides for target language",
        "Include error handling and edge case consideration",
        "Provide time/space complexity analysis",
        "Suggest alternative implementations when applicable",
        "Add integration notes for different environments"
    ],
    add_datetime_to_instructions=True,
)

reasoner = Agent(
    name="Analytical Strategist",
    role="Complex Problem Solving Expert",
    model=Ollama(id=reasoner_llm),
    description=(
        "Expert in mathematical modeling and logical deduction. "
        "Specializes in multi-step reasoning, probability analysis, "
        "and systems thinking for complex decision-making."
    ),
    instructions=[
        "Break problems into first principles",
        "Validate assumptions through multiple approaches",
        "Quantify uncertainty in conclusions",
        "Provide visualizable mental models",
        "Highlight potential cognitive biases"
    ],
    add_datetime_to_instructions=True,
)

# Define type aliases for clarity
AgentResponseDict = Dict[str, Any]
SummaryStructure = Dict[str, str]

def process_subqueries(
    sub_queries: List[SubQuery], 
    agents: Dict[int, Agent]
) -> AgentResponseDict:
    """Process subqueries using appropriate agents with error handling."""
    responses: AgentResponseDict = {}
    
    category_names = {
        0: "search_results",
        1: "code_solutions",
        2: "analysis"
    }

    for sub_query in sub_queries:
        try:
            agent = agents.get(sub_query.category)
            if not agent:
                raise ValueError(f"No agent for category {sub_query.category}")

            response = agent.run(sub_query.sub_query)
            responses.setdefault(category_names[sub_query.category], {})[sub_query.sub_query] = {
                "content": response.content,
                "metadata": {
                    "source": agent.name
                }
            }
            pprint_run_response(response, markdown=True)
            
        except Exception as e:
            responses.setdefault("errors", []).append({
                "sub_query": sub_query.sub_query,
                "error": str(e)
            })
            print(f"Error processing '{sub_query.sub_query}': {e}")
    
    return responses

def generate_summary(responses: AgentResponseDict, summarizer: Agent) -> SummaryStructure:
    """Generate structured summary from collected responses."""
    summary_prompt = f"""
    Synthesize key information from these research components:
    
    1. Search Results: {responses.get('search_results', {})}
    2. Technical Solutions: {responses.get('code_solutions', {})}
    3. Analytical Insights: {responses.get('analysis', {})}
    
    Create a structured summary with:
    - Key findings from credible sources
    - Recommended technical approaches
    - Critical insights from analysis
    - Potential limitations or contradictions
    """
    
    try:
        summary_response: RunResponse = summarizer.run(summary_prompt)
        return summary_response
    except Exception as e:
        return {"summary_error": str(e)}

# Main execution flow
final_response = {
    "original_query": query,
    "research_components": process_subqueries(
        response.content.sub_queries,
        agents={0: searcher, 1: coder, 2: reasoner}
    )
}

# Configure summarizer with enhanced capabilities
summarizer = Agent(
    name="Research Synthesizer",
    model=Ollama(id=summarizer_llm),
    description=(
        "Expert in multi-source synthesis and contradiction resolution. "
        "Specializes in integrating technical, analytical, and empirical data "
        "into actionable intelligence reports."
    ),
    instructions=[
        "Identify consensus and contradictions between sources",
        "Highlight statistically significant findings",
        "Prioritize recent information (within 2 years unless historical context needed)",
        "Maintain traceability to source materials",
        "Include confidence estimates for key claims"
    ],
    add_datetime_to_instructions=True,
)

# Generate and store final summary
summary_response = generate_summary(final_response["research_components"], summarizer)
pprint("Final Summary:")
pprint_run_response(summary_response, markdown=True)