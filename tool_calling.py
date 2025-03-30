from langchain_community.llms import Ollama
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import render_text_description
from typing import Any, Dict, Optional, TypedDict
from langchain_core.runnables import RunnableConfig
from langchain_community.tools import DuckDuckGoSearchResults

# Initialize models
reasoner = Ollama(model="exaone-deep:2.4b")
coder = Ollama(model="qwen2.5-coder:3b")
qwen = Ollama(model="qwen2.5:3b")  # Tool-calling model

@tool
def search(query: str) -> str:
    """Search the web for current information. Returns 3 results."""
    return str(DuckDuckGoSearchResults(max_results=3).run(query))


# Engineer tool handling (using Qwen)
engineer_tools = [search]
engineer_prompt = ChatPromptTemplate.from_template(
    """You're an engineering assistant. Choose between these tools:
    
    Tools:
    {tools}
    
    User question: {input}
    
    Respond with JSON containing 'name' (tool name) and 'arguments' (key-value pairs)."""
)
engineer_chain = engineer_prompt | qwen | JsonOutputParser()

@tool
def engineer_assistant(query: str) -> str:
    """Handle search and calculation requests. Use for general knowledge or math problems."""
    try:
        # Get tool selection from Qwen
        tool_call = engineer_chain.invoke({
            "input": query,
            "tools": render_text_description(engineer_tools)
        })
        
        # Execute the selected tool
        if tool_call["name"] == "search":
            return search.invoke(tool_call["arguments"]["query"])
        elif tool_call["name"] == "calculate":
            return calculate.invoke(tool_call["arguments"]["expression"])
        return "No valid tool found"
    except Exception as e:
        return f"Engineering error: {str(e)}"

@tool
def code_answer(query: str) -> str:
    """Answer programming and code-related questions."""
    return coder.invoke(query)

@tool
def reason_answer(query: str) -> str:
    """Solve complex mathematical and logical problems."""
    return reasoner.invoke(query)

# Main tool set
tools = [reason_answer, code_answer, engineer_assistant]
rendered_tools = render_text_description(tools)

system_prompt = f"""\
You are an assistant with these capabilities:

{rendered_tools}

Choose the best tool and respond with JSON containing 'name' and 'arguments'.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}")
])

class ToolCallRequest(TypedDict):
    name: str
    arguments: Dict[str, Any]

def invoke_tool(tool_call: ToolCallRequest, config: Optional[RunnableConfig] = None):
    tool_map = {tool.name: tool for tool in tools}
    return tool_map[tool_call["name"]].invoke(tool_call["arguments"], config=config)

chain = prompt | Ollama(model="gemma3:4b") | JsonOutputParser() | invoke_tool

# Test the system
# print(chain.invoke({"input": "What's currently happening in France?"}))
print(chain.invoke({"input": "Calculate 15 * 24 + 30"}))
# print(chain.invoke({"input": "Write a Python Fibonacci function"}))