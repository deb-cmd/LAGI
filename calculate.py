from langchain_community.llms import Ollama
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import render_text_description
from langchain_core.output_parsers import JsonOutputParser
from typing import Any, Dict, Optional, TypedDict
from langchain_core.runnables import RunnableConfig

model = Ollama(model="gemma3:4b")

@tool
def multiply(x: float, y: float) -> float:
    """Multiply two numbers together."""
    return x * y


@tool
def add(x: int, y: int) -> int:
    "Add two numbers."
    return x + y


tools = [multiply, add]
    
rendered_tools = render_text_description(tools)

system_prompt = f"""\
You are an assistant that has access to the following set of tools. 
Here are the names and descriptions for each tool:

{rendered_tools}

Given the user input, return the name and input of the tool to use. 
Return your response as a JSON blob with 'name' and 'arguments' keys.

The `arguments` should be a dictionary, with keys corresponding 
to the argument names and the values corresponding to the requested values.
"""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{input}")]
)

class ToolCallRequest(TypedDict):
    """A typed dict that shows the inputs into the invoke_tool function."""

    name: str
    arguments: Dict[str, Any]

def invoke_tool(
    tool_call_request: ToolCallRequest, config: Optional[RunnableConfig] = None
):
    """A function that we can use the perform a tool invocation.

    Args:
        tool_call_request: a dict that contains the keys name and arguments.
            The name must match the name of a tool that exists.
            The arguments are the arguments to that tool.
        config: This is configuration information that LangChain uses that contains
            things like callbacks, metadata, etc.See LCEL documentation about RunnableConfig.

    Returns:
        output from the requested tool
    """
    tool_name_to_tool = {tool.name: tool for tool in tools}
    name = tool_call_request["name"]
    requested_tool = tool_name_to_tool[name]
    return requested_tool.invoke(tool_call_request["arguments"], config=config)

chain = prompt | model | JsonOutputParser() | invoke_tool
message = chain.invoke({"input": "what's thirteen times 4.14137281"})
print(message)