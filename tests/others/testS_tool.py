from max_ai.tools.function_as_tool import FunctionAsTool
import json

async def search_docs(query: str, limit: int = 5) -> list[str]:
    """Search documentation.
    
    Args:
        query: What to search.
        limit: Max results.
    """
    return []

tool = FunctionAsTool(search_docs)
print(json.dumps(tool.parameters, indent=2))