from __future__ import annotations as _annotations

from dataclasses import dataclass
import logging
import os
from typing import List

import aiohttp
import asyncio

from pprint import pformat

from pydantic_ai import Agent, ModelRetry, RunContext, settings
from pydantic_ai.models.ollama import OllamaModel
from supabase import create_client, Client

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -------------------------------------
# Set up logging
# Define log format with color and emoji
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

# Initialize Supabase client
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
            'get_page_content'
        ]

    def track_tool(self, tool_name: str):
        self.tools_used.append(tool_name)
        logging.debug(f"Tool used: {tool_name} ✅")

    def get_missing_tools(self) -> List[str]:
        return [tool for tool in self.required_tools if tool not in self.tools_used]
tool_tracker = ToolUsageTracker()

def update_system_prompt(attempt: int) -> str:
    base_prompt = """
You are an expert at Pydantic AI - a Python AI agent framework that you have access to all the documentation to,
including examples, an API reference, and other resources to help you build Pydantic AI agents, use it to answer users question.

Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering the user's question unless you have already.

When you first look at the documentation, always start with RAG.
Then also always check the list of available documentation pages and retrieve the content of page(s) if it'll help.

Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
To ensure accuracy and efficiency, always consult the Pydantic AI documentation and follow this structured approach:

1. **Retrieve Relevant Documents**: Use `retrieve_relevant_documentation` to get the most relevant documentation chunks based on the user's query.
2. **Analyze Documentation Pages**: Use `list_documentation_pages` to identify all available documentation pages that could help answer the question.
3. **Generate Content Summary**: Use `get_page_content` to gather and summarize the full content of any relevant documentation pages identified.

"""

    if attempt > 0:
        unused_tools = tool_tracker.get_missing_tools()
        additional_instructions = f"""
        IMPORTANT: In your previous {attempt} attempt{'s' if attempt > 1 else ''}, you failed to use all required tools or provide a complete answer.
        It is CRUCIAL that you use ALL tools in the specified order before answering.
        Failure to do so will result in an incorrect response.
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
                    logging.error(f"Error getting embedding: HTTP {response.status}")
                    return [0] * 768  # Changed to 768 for nomic-embed-text
    except Exception as e:
        logging.error(f"Error getting embedding: {e}")
        return [0] * 768  # Changed to 768 for nomic-embed-text

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
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query)
        # Query Supabase for relevant documents
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
            
        # Format the results
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
        # Query Supabase for unique URLs where source is pydantic_ai_docs
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .execute()
        
        if not result.data:
            return []
            
        # Extract unique URLs
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
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        return f"Error retrieving page content: {str(e)}"

async def validate_answer_against_query( user_query: str, generated_answer: str) -> bool:
    """
    Validate if the generated answer matches the intent of the user's query using Pydantic AI.
    
    Args:
        ctx: The context containing the Pydantic AI agent
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
        # logging.info(f"validation result: {result}")
        return result.data
        
    except Exception as e:
        logging.error(f"Error validating answer: {e}")
        return False

async def check_database_content():
    result = supabase.from_('site_pages').select('*').execute()
    if len(result.data) > 0:
        logging.info(f"Total documents in database: {len(result.data)}; Sample document found")

def format_list_for_logging(items):
    return "\n" + pformat(items, indent=2)

async def main():
    logging.info("Starting main function")

    await check_database_content()
    # original_question = "what models are supported by PydanticAI?"
    original_question = "get me the Weather Agent Example"
    
    tool_instructions = """
    You MUST use these tools in the following order:
    1. retrieve_relevant_documentation
    2. list_documentation_pages
    3. get_page_content

    Do not skip any steps. After using all tools, provide your final answer.
    """

    for attempt in range(3):
        logging.info(f"Attempt {attempt + 1}")
        tool_tracker.tools_used = []
            # Update the system prompt based on previous attempts
        updated_system_prompt = update_system_prompt(attempt)
        logging.info(f"system_prompt to be executed: {updated_system_prompt}")
        deps = PydanticAIDeps(supabase=supabase,
                            model=model,
                            system_prompt=updated_system_prompt
                            )
        # Prepare the full prompt
        # full_prompt = f"Question: {original_question}"
        
        # # If it's not the first attempt, add a reminder about unused tools
        # if attempt > 0:
        #     unused_tools = tool_tracker.get_missing_tools()
        #     if unused_tools:
        #         tool_reminder = f"In your previous attempt, you didn't use these tools: {', '.join(unused_tools)}. Please ensure you use ALL tools in the correct order before providing an answer."
        #         full_prompt = f"Question: {original_question} "

        # Run the agent with the full prompt
        pydantic_ai_expert.model_settings = settings.ModelSettings(
            temperature=0.0,
            parallel_tool_calls=False
        )
        logging.info(f"prompt to be executed: {original_question}")
        response = await pydantic_ai_expert.run(user_prompt=original_question, deps=deps)
        logging.info(f"before validation answer: {response.data}")
        # Check which tools were used
        unused_tools = tool_tracker.get_missing_tools()
        logging.info(f"Tools used in attempt {attempt + 1}: {tool_tracker.tools_used}")
        logging.info(f"Unused tools: {unused_tools}")
        logging.info(f"pydantic_ai_expert response: {response.data}")
        # Validate the answer
        validation_result = await validate_answer_against_query(
            user_query=original_question,
            generated_answer=str(response.data),
        )

        if str(validation_result).lower().startswith('yes') and len(unused_tools) < 1:
            logging.info(f"Answer validated and all tools used on attempt {attempt + 1}")
            break
        elif str(validation_result).lower().startswith('yes') and len(unused_tools) > 0:
            logging.warning("Answer validated but not all tools were used. Retrying.")
        else:
            logging.warning(f"Validation failed on attempt {attempt + 1},\n Reason: {validation_result}")

    else:
        logging.error("Failed to get a valid answer using all tools after 3 attempts")
        response.data = "Unable to generate a valid answer using all required tools after multiple attempts. Please try rephrasing your question."

    logging.info(f"Final Answer: {response.data}")
if __name__ == "__main__":
    asyncio.run(main())