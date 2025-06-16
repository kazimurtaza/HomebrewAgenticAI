from __future__ import annotations as _annotations

from dataclasses import dataclass
import logging
import os
from typing import List, Optional, Dict, Any

import aiohttp
import asyncio
from pprint import pformat

from pydantic_ai import Agent, RunContext, settings
from pydantic_ai.models.ollama import OllamaModel
from supabase import create_client, Client
from mem0 import MemoryClient

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Logging Setup ---
LOG_FORMAT = "{}%(asctime)s - %(levelname)s - %(message)s"

class ColoredEmojiFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: LOG_FORMAT.format("🔍 "),
        logging.INFO: LOG_FORMAT.format("🔍 "),
        logging.WARNING: LOG_FORMAT.format("⚠️ "),
        logging.ERROR: LOG_FORMAT.format("❌ "),
        logging.CRITICAL: LOG_FORMAT.format("🚨 ")
    }
    LEVEL_COLORS = {
        logging.DEBUG: "\033[90m",
        logging.INFO: "\033[92m",
        logging.WARNING: "\033[93m",
        logging.ERROR: "\033[91m",
        logging.CRITICAL: "\033[91m"
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        formatted_message = formatter.format(record)
        color = self.LEVEL_COLORS.get(record.levelno)
        return f"{color}{formatted_message}\033[0m"

def setup_logging():
    logging.basicConfig(level=os.getenv('LOG_LEVEL', 'DEBUG'))
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredEmojiFormatter())
    logger.addHandler(console_handler)

setup_logging()
# -------------------------------------

# Initialize clients
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
model = OllamaModel(model_name=os.getenv('OLLAMA_MODEL_NAME'), base_url=os.getenv('OLLAMA_BASE_URL'))
mem0_client = MemoryClient()

@dataclass
class PydanticAIDeps:
    supabase: Client
    model: OllamaModel
    system_prompt: str

pydantic_ai_expert = Agent(model, deps_type=PydanticAIDeps)

# --- Tool Definitions ---

async def get_embedding(text: str) -> List[float]:
    url = os.getenv('EMBEDDING_API_URL', 'http://127.0.0.1:11434/api/embeddings')
    payload = {"model": os.getenv('EMBEDDING_MODEL', 'nomic-embed-text'), "prompt": text}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                return result.get('embedding', [])
    except Exception as e:
        logging.error(f"Error getting embedding: {e}")
        return []

@pydantic_ai_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    logging.info(f"TOOL: Retrieving docs for query: '{user_query}'")
    try:
        query_embedding = await get_embedding(user_query)
        if not query_embedding: return "Error creating embedding for query."
        result = ctx.deps.supabase.rpc('match_site_pages', {'query_embedding': query_embedding, 'match_count': 5, 'filter': {}}).execute()
        if not result.data: return "No relevant documentation found."
        return "\n\n---\n\n".join([f"# {doc['title']}\nURL: {doc['url']}\n\n{doc['content']}" for doc in result.data])
    except Exception as e:
        logging.error(f"Error retrieving documentation: {e}")
        return f"Error: {str(e)}"

@pydantic_ai_expert.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    logging.info("TOOL: Listing all documentation pages.")
    try:
        result = ctx.deps.supabase.from_('site_pages').select('url', count='exact').execute()
        return sorted(list(set(doc['url'] for doc in result.data)))
    except Exception as e:
        logging.error(f"Error listing documentation pages: {e}")
        return []

@pydantic_ai_expert.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    logging.info(f"TOOL: Getting content for EXACT URL: '{url}'")
    try:
        result = ctx.deps.supabase.from_('site_pages').select('title, content, chunk_number').eq('url', url).order('chunk_number').execute()
        if not result.data: return f"No content found for URL: {url}"
        page_title = result.data[0]['title'].split(' - ')[0]
        return f"# {page_title}\n\n" + "\n\n".join([chunk['content'] for chunk in result.data])
    except Exception as e:
        logging.error(f"Error retrieving page content: {e}")
        return f"Error: {str(e)}"

# --- Stateful Workflow Runner ---

async def run_rag_workflow(original_question: str) -> str:
    deps = PydanticAIDeps(supabase=supabase, model=model, system_prompt="")

    # --- Step 1: Retrieve initial documents ---
    logging.info("--- WORKFLOW STEP 1: Retrieving initial documents ---")
    deps.system_prompt = "You MUST call the `retrieve_relevant_documentation` tool with the user's original query. The `user_query` is the original question."
    retrieved_docs_result = await pydantic_ai_expert.run(user_prompt=original_question, deps=deps)
    retrieved_docs = retrieved_docs_result.data
    logging.info(f"Step 1 Result (Retrieved Docs): {retrieved_docs}...")

    # --- Step 2: List all documentation pages for debugging ---
    logging.info("--- WORKFLOW STEP 2: Listing all pages for context ---")
    deps.system_prompt = "You MUST call the `list_documentation_pages` tool. It takes no arguments."
    all_pages_result = await pydantic_ai_expert.run(user_prompt="List all documentation pages.", deps=deps)
    all_pages = all_pages_result.data
    
    # --- NEW: Diagnostic Logging ---
    # Log a few of the URLs from the database to see their exact format.
    if all_pages:
        logging.info(f"Step 2 DIAGNOSTIC: Sample URLs from DB: {pformat(all_pages)}")
    else:
        logging.warning("Step 2 DIAGNOSTIC: The 'list_documentation_pages' tool returned no URLs.")


    # --- Step 3: Get the content of the most relevant page ---
    logging.info("--- WORKFLOW STEP 3: Getting specific page content ---")
    step3_prompt = f"""
    The following documentation was retrieved based on the user's query:
    ---
    {retrieved_docs}
    ---
    From the documentation above, which includes URLs, identify the single most relevant URL and call the `get_page_content` tool with that EXACT URL string.
    """
    deps.system_prompt = "You MUST call the `get_page_content` tool with the most relevant URL from the provided context."
    specific_page_content_result = await pydantic_ai_expert.run(user_prompt=step3_prompt, deps=deps)
    specific_page_content = specific_page_content_result.data
    logging.info(f"Step 3 Result (Specific Content): {specific_page_content[:300]}...")

    # --- Step 4: Synthesize the final answer ---
    logging.info("--- WORKFLOW STEP 4: Synthesizing the final answer ---")
    final_prompt = f"""
    Based on ALL the information gathered below, please provide a comprehensive answer to the user's original question: '{original_question}'

    **Initial Document Chunks:**
    ---
    {retrieved_docs}
    ---

    **Specific Page Content:**
    ---
    {specific_page_content}
    ---

    Now, synthesize this information into a clear and comprehensive final answer.
    """
    deps.system_prompt = "You are a helpful assistant. Synthesize the provided information into a clear and comprehensive final answer."
    final_answer_result = await pydantic_ai_expert.run(user_prompt=final_prompt, deps=deps)
    
    return final_answer_result.data

async def main():
    original_question = "get me the Weather Agent Example"
    logging.info(f"--- Starting Agent Workflow for query: '{original_question}' ---")
    final_answer = await run_rag_workflow(original_question)
    #logging.info(f"\n--- Final Answer ---\n{final_answer}")

if __name__ == "__main__":
    asyncio.run(main())