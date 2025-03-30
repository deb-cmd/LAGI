from agno.agent import Agent, RunResponse
from agno.models.ollama import Ollama
from inference import QueryClassifier  # Your custom classifier
from agno.utils.pprint import pprint_run_response
# Initialize the query classifier
classifier = QueryClassifier()

def run_ollama_model(query: str, model_id: str = "qwen2.5:3b") -> RunResponse:
    """Run a query through Ollama with specified parameters."""
    agent = Agent(
        model=Ollama(
            id=model_id,
        ),
        markdown=True
    )
    return agent.run(query)

# Example usage with classification and model routing
def process_query(query: str):
    # Classify the query
    category = classifier.predict(query)
    
    # Route to appropriate model
    if category == 'code':
        response = run_ollama_model(query, model_id="qwen2.5-coder:3b")
    elif category == 'language':
        response = run_ollama_model(query, model_id="qwen2.5:3b")
    elif category == 'reason':
        response = run_ollama_model(query, model_id="exaone-deep:2.4b")
    else:
        response = "unknown category"
    
    return response

# Test the pipeline
if __name__ == "__main__":
    
    # Example queries
    test_queries = [
        "Explain quantum entanglement in simple terms",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = process_query(query)
        pprint_run_response(result, markdown=True)