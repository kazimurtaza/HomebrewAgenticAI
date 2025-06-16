from __future__ import annotations as _annotations

from dataclasses import dataclass
import logging
import os
import json
from typing import List, Optional, Dict, Any

import aiohttp
import asyncio
import httpx 

from pprint import pformat

from pydantic_ai import Agent, ModelRetry, RunContext, settings
from pydantic_ai.models.ollama import OllamaModel
from supabase import create_client, Client

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Logging Setup (Keep as is) ---
LOG_FORMAT = "{}%(asctime)s - %(levelname)s - %(message)s"

class ColoredEmojiFormatter(logging.Formatter):
    """Formatter that adds color and emoji for different log levels."""
    
    FORMATS = {
        logging.DEBUG: LOG_FORMAT.format("🔍 "),
        logging.INFO: LOG_FORMAT.format("🔍 "),
        logging.WARNING: LOG_FORMAT.format("⚠️ "),
        logging.ERROR: LOG_FORMAT.format("❌ "),
        logging.CRITICAL: LOG_FORMAT.format("🚨 ")
    }

    LEVEL_COLORS = {
        logging.DEBUG: "\033[90m",  # Gray
        logging.INFO: "\033[92m",   # Green
        logging.WARNING: "\033[93m",  # Orange
        logging.ERROR: "\033[91m",   # Red
        logging.CRITICAL: "\033[91m"  # Red
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        formatted_message = formatter.format(record)
        color = self.LEVEL_COLORS.get(record.levelno)
        return f"{color}{formatted_message}\033[0m"  # Reset color after message

def setup_logging():
    # Set up basic configuration
    logging.basicConfig(level=os.getenv('LOG_LEVEL', 'DEBUG'))
    
    # Get the root logger
    logger = logging.getLogger()
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler with custom formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredEmojiFormatter())
    
    # Add console handler to the logger
    logger.addHandler(console_handler)

# Call this function at the start of your script
setup_logging()
# -------------------------------------

# Initialize Supabase client (kept as is, as per your request to keep direct Supabase RAG)
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Initialize Ollama model
model = OllamaModel(
    model_name=os.getenv('OLLAMA_MODEL_NAME'),
    base_url=os.getenv('OLLAMA_BASE_URL')
)
validationModel = OllamaModel(
    model_name="deepseek-r1:14b",
    base_url=os.getenv('OLLAMA_BASE_URL')
)

@dataclass
class PydanticAIDeps:
    supabase: Client
    model: OllamaModel
    system_prompt: str

@dataclass
class RetrievalResult:
    content: str
    debug_info: dict

class ToolUsageTracker:
    def __init__(self):
        self.tools_used = []
        self.required_tools = [
            'retrieve_relevant_documentation',
            'list_documentation_pages',
            'get_page_content',
            'add_memory',
            'search_memory',
            'list_memories',
        ]

    def track_tool(self, tool_name: str):
        self.tools_used.append(tool_name)
        logging.debug(f"Tool used: {tool_name} ✅")

    def get_missing_tools(self) -> List[str]:
        return [tool for tool in self.required_tools if tool not in self.tools_used]
tool_tracker = ToolUsageTracker()

def update_system_prompt(attempt: int) -> str:
    base_prompt = """
You are an expert AI assistant designed to help users with information using various tools.
Your primary job is to assist the user by utilizing the tools provided.

Do not ask the user before taking an action, just do it.

Available Tools:
- `retrieve_relevant_documentation`: Get documentation chunks semantically relevant to a query from the Pydantic AI docs. Use this to start your research.
- `list_documentation_pages`: Get a list of all available documentation page URLs for Pydantic AI. Use this to discover specific pages.
- `get_page_content`: Retrieve the full content of a specific documentation page by its URL. Use this to get detailed information from a page.
- `add_memory`: Store important information or conversation context in long-term memory.
- `search_memory`: Retrieve relevant memories based on a query.
- `list_memories`: View all stored memories.

When responding, always prioritize using documentation tools (`retrieve_relevant_documentation`, `list_documentation_pages`, `get_page_content`) first if the question is about Pydantic AI documentation.
Use memory tools (`add_memory`, `search_memory`, `list_memories`) to remember user preferences, conversation history, or new insights gained during your research.
If you gain new valuable information or insights from the documentation or user interactions, remember to store them using `add_memory`.
"""

    if attempt > 0:
        unused_tools = tool_tracker.get_missing_tools()
        if unused_tools:
            additional_instructions = f"""
            IMPORTANT: In your previous {attempt} attempt{'s' if attempt > 1 else ''}, you failed to use all required tools or provide a complete answer.
            It is CRUCIAL that you use ALL tools that are relevant to the user's request, especially those related to documentation retrieval if the question requires it.
            Failure to do so will result in an incorrect response.
            In your previous attempt, you didn't use these tools: {', '.join(unused_tools)}.
            """
            base_prompt += additional_instructions

    return base_prompt

validation_system_prompt = """
        You are an expert at validating responses to ensure they accurately address the original query.
        Your primary objective is to verify, if the answer matches the question.
        Answer with "Yes" if it matches, otherwise "The answer does not relate to the question because".

        Examples of responses:

        - Yes: "The answer properly addresses the query."

        - No: "The answer does not relate to the question."

        Only return "Yes" or "This is not accurate because," - additional explanation.
        """

pydantic_ai_expert = Agent(
    model,
    deps_type=PydanticAIDeps,
    retries=5
)

# --- NEW: MCP Client Utility for Mem0.ai ---
async def call_mcp_server(server_url: str, tool_name: str, **kwargs: Any) -> Any:
    """
    Generic asynchronous function to call an MCP server tool.
    Assumes the MCP server exposes tools via a POST endpoint like /{tool_name}.
    It expects a JSON response.
    """
    full_url = f"{server_url}/{tool_name}"
    headers = {"Content-Type": "application/json"}
    payload = kwargs # The arguments to the tool become the JSON payload

    logging.info(f"Calling MCP tool: {tool_name} at {full_url} with payload: {payload}")

    try:
        async with httpx.AsyncClient() as client: # Use httpx for client calls
            response = await client.post(full_url, headers=headers, json=payload, timeout=60.0)
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            result = response.json()
            logging.info(f"MCP tool {tool_name} response: {result}")
            return result.get('result', result) # Return 'result' key if exists, else the whole response
    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP error calling MCP tool {tool_name}: {e.response.status_code} - {e.response.text}")
        return f"Error calling MCP tool {tool_name}: {e.response.text}"
    except httpx.RequestError as e:
        logging.error(f"Network error calling MCP tool {tool_name}: {e}")
        return f"Network error calling MCP tool {tool_name}: {e}"
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error from MCP tool {tool_name}: {e} - Response: {response.text}")
        return f"JSON decode error from MCP tool {tool_name}: {e}"
    except Exception as e:
        logging.error(f"An unexpected error occurred calling MCP tool {tool_name}: {e}")
        return f"An unexpected error occurred calling MCP tool {tool_name}: {e}"

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from Ollama API."""
    url = os.getenv('EMBEDDING_API_URL', 'http://10.0.0.12:11434/api/embeddings')
    payload = {
        "model": os.getenv('EMBEDDING_MODEL', 'nomic-embed-text'),
        "prompt": text
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['embedding']
                else:
                    logging.error(f"Error getting embedding from Ollama: HTTP {response.status} - {response.text}")
                    return [0] * 768
    except Exception as e:
        logging.error(f"Error getting embedding from Ollama: {e}")
        return [0] * 768


# --- Existing Supabase-backed Tools (Retained and unchanged from your provided code) ---
@pydantic_ai_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and Ollama client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    tool_tracker.track_tool("retrieve_relevant_documentation")
    try:
        query_embedding = await get_embedding(user_query)
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {'source': 'pydantic_ai_docs'}
            }
        ).execute()
        
        if not result.data:
            return "No relevant documentation found."
            
        formatted_chunks = [
            f"# {doc['title']}\n\n{doc['content']}"
            for doc in result.data
        ]
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@pydantic_ai_expert.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    tool_tracker.track_tool("list_documentation_pages")
    try:
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .execute()
        
        if not result.data:
            return []
            
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@pydantic_ai_expert.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    tool_tracker.track_tool("get_page_content")
    try:
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        page_title = result.data[0]['title'].split(' - ')[0]
        formatted_content = [f"# {page_title}\n"]
        
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        return f"Error retrieving page content: {str(e)}"

# --- Mem0.ai Tools (New additions) ---
@pydantic_ai_expert.tool
async def add_memory(ctx: RunContext[PydanticAIDeps], content: str, user_id: Optional[str] = None, session_id: Optional[str] = None) -> str:
    """
    Stores a piece of text content as a memory for the RAG agent.
    
    Args:
        ctx: The context (unused directly for MCP call)
        content: The text content to store as memory.
        user_id: Optional user ID to associate the memory with (defaults to DEFAULT_USER_ID from .env).
        session_id: Optional session ID to associate the memory with.
        
    Returns:
        str: Confirmation message from Mem0.ai.
    """
    tool_tracker.track_tool("add_memory")
    mem0_mcp_server_url = f"http://10.0.0.5:{os.getenv('MEM0_MCP_HOST_PORT')}"
    
    user_id = user_id if user_id else os.getenv('DEFAULT_USER_ID', 'default_rag_user')
    
    payload = {
        "content": content,
        "userId": user_id,
    }
    if session_id:
        payload["sessionId"] = session_id
        
    mcp_response = await call_mcp_server(
        mem0_mcp_server_url, 
        "add_memory",
        **payload
    )
    
    if isinstance(mcp_response, str) and "Error" in mcp_response:
        return f"Failed to add memory: {mcp_response}"
    
    return f"Memory added successfully for user {user_id}. Memory ID: {mcp_response.get('memoryId', 'N/A')}"

@pydantic_ai_expert.tool
async def search_memory(ctx: RunContext[PydanticAIDeps], query: str, user_id: Optional[str] = None, session_id: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Semantically searches stored memories for relevant information.
    
    Args:
        ctx: The context (unused directly for MCP call)
        query: The natural language query to search memories with.
        user_id: Optional user ID to filter memories by (defaults to DEFAULT_USER_ID from .env).
        session_id: Optional session ID to filter memories by.
        limit: Maximum number of relevant memories to retrieve.
        
    Returns:
        List[Dict[str, Any]]: A list of relevant memory objects.
    """
    tool_tracker.track_tool("search_memory")
    mem0_mcp_server_url = f"http://10.0.0.5:{os.getenv('MEM0_MCP_HOST_PORT')}"
    
    user_id = user_id if user_id else os.getenv('DEFAULT_USER_ID', 'default_rag_user')

    payload = {
        "query": query,
        "userId": user_id,
        "limit": limit
    }
    if session_id:
        payload["sessionId"] = session_id
        
    mcp_response = await call_mcp_server(
        mem0_mcp_server_url, 
        "search_memory",
        **payload
    )
    
    if isinstance(mcp_response, str) and "Error" in mcp_response:
        logging.error(f"Failed to search memory: {mcp_response}")
        return []
    
    return mcp_response.get('results', [])

@pydantic_ai_expert.tool
async def list_memories(ctx: RunContext[PydanticAIDeps], user_id: Optional[str] = None, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Lists all stored memories for a given user or session.
    
    Args:
        ctx: The context (unused directly for MCP call)
        user_id: Optional user ID to filter memories by (defaults to DEFAULT_USER_ID from .env).
        session_id: Optional session ID to filter memories by.
        
    Returns:
        List[Dict[str, Any]]: A list of all stored memory objects.
    """
    tool_tracker.track_tool("list_memories")
    mem0_mcp_server_url = f"http://10.0.0.5:{os.getenv('MEM0_MCP_HOST_PORT')}"
    
    user_id = user_id if user_id else os.getenv('DEFAULT_USER_ID', 'default_rag_user')

    payload = {
        "userId": user_id,
    }
    if session_id:
        payload["sessionId"] = session_id
        
    mcp_response = await call_mcp_server(
        mem0_mcp_server_url, 
        "list_memories",
        **payload
    )
    
    if isinstance(mcp_response, str) and "Error" in mcp_response:
        logging.error(f"Failed to list memories: {mcp_response}")
        return []
    
    return mcp_response.get('results', [])


async def validate_answer_against_query( user_query: str, generated_answer: str) -> bool:
    """
    Validate if the generated answer matches the intent of the user's query using Pydantic AI.
    
    Args:
        user_query: The original user query
        generated_answer: The answer generated by the system
        
    Returns:
        bool: True if the answer matches the query intent, False otherwise
    """
    try:
        validation_prompt = f"Does this answer properly address the following question? Answer with 'Yes' or 'No'.\n\nQuestion: {user_query}\nAnswer: {generated_answer}"
        validation_agent = Agent(model=validationModel, system_prompt=validation_system_prompt)
        validation_agent.model_settings = settings.ModelSettings(
            temperature=0.0,
            parallel_tool_call=False,
        )
        result = await validation_agent.run(
            user_prompt=validation_prompt
        )
        return str(result.data).lower().startswith('yes')
        
    except Exception as e:
        logging.error(f"Error validating answer: {e}")
        return False

async def check_database_content():
    # This function uses the direct Supabase client, as per your request to keep it for RAG.
    try:
        result = supabase.from_('site_pages').select('*').execute()
        if len(result.data) > 0:
            logging.info(f"Total documents in database (direct Supabase): {len(result.data)}; Sample document found")
        else:
            logging.info("No documents found in Supabase database (direct Supabase check).")
    except Exception as e:
        logging.error(f"Error checking database content directly via Supabase: {e}")
        logging.warning("Ensure your Supabase is running and accessible from this script's environment.")


def format_list_for_logging(items):
    return "\n" + pformat(items, indent=2)

async def main():
    logging.info("Starting main function with Mem0.ai MCP integration and direct Supabase RAG.")

    # Initial check for database content (still using direct Supabase as per this version's setup)
    await check_database_content()

    # Example of adding an initial memory
    logging.info("Attempting to add an initial memory to Mem0.ai.")
    await add_memory(None, "The user is interested in agentic RAG, Pydantic AI documentation, and using memory capabilities.", user_id=os.getenv('DEFAULT_USER_ID', 'default_rag_user'), session_id="initial_session")
    
    # Example of searching memories
    logging.info("Attempting to search memories.")
    search_results = await search_memory(None, "what is this conversation about?", user_id=os.getenv('DEFAULT_USER_ID', 'default_rag_user'), session_id="initial_session", limit=2)
    logging.info(f"Search results from memory: {search_results}")
    
    original_question = "What models are supported by PydanticAI and can you also remember that I prefer short, direct answers?"
    # You can change the question to test different tool usages:
    # original_question = "Retrieve documentation about the 'Agent' class in PydanticAI."
    # original_question = "List all memories you have for this session."


    for attempt in range(3):
        logging.info(f"Attempt {attempt + 1}")
        tool_tracker.tools_used = []
        updated_system_prompt = update_system_prompt(attempt)
        logging.info(f"system_prompt to be executed: {updated_system_prompt}")
        
        deps = PydanticAIDeps(
            supabase=supabase, # Pass the direct Supabase client as dependency for direct Supabase RAG tools
            model=model,
            system_prompt=updated_system_prompt
        )
        
        pydantic_ai_expert.model_settings = settings.ModelSettings(
            temperature=0.0,
            parallel_tool_calls=False # Keep parallel_tool_calls to False for strict ordering logic
        )
        
        logging.info(f"prompt to be executed: {original_question}")
        response = await pydantic_ai_expert.run(user_prompt=original_question, deps=deps)
        
        logging.info(f"before validation answer: {response.data}")
        
        # Check which tools were used
        unused_tools = tool_tracker.get_missing_tools()
        logging.info(f"Tools used in attempt {attempt + 1}: {tool_tracker.tools_used}")
        logging.info(f"Unused tools: {unused_tools}")
        logging.info(f"pydantic_ai_expert response: {response.data}")
        
        validation_result = await validate_answer_against_query(
            user_query=original_question,
            generated_answer=str(response.data),
        )

        if validation_result and len(unused_tools) < 1:
            logging.info(f"Answer validated and all relevant tools used on attempt {attempt + 1}")
            break
        elif validation_result and len(unused_tools) > 0:
            logging.warning("Answer validated but not all relevant tools were used. Retrying.")
        else:
            logging.warning(f"Validation failed on attempt {attempt + 1},\n Reason: {validation_result}")

    else:
        logging.error("Failed to get a valid answer after 3 attempts. Please try rephrasing your question.")
        response.data = "Unable to generate a valid answer after multiple attempts. Please try rephrasing your question."

    logging.info(f"Final Answer: {response.data}")

if __name__ == "__main__":
    asyncio.run(main())
