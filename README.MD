# 🐍 HomebrewAgenticAI - Poor Man's Agentic RAG
This project implements a "Poor Man's Agentic RAG (Retrieval Augmented Generation)" system in Python, inspired by the concepts discussed in the blog "Building a 'Poor Man's Agentic RAG' with Python: A Step-by-Step Guide". It combines a web crawler for data ingestion, a local Large Language Model (LLM) for summarization and embeddings, and a Supabase instance for vector storage and retrieval. The system also includes an agentic loop with a self-correction mechanism to improve the quality of generated answers by ensuring proper tool usage.

# ✨ Features
- Documentation Crawling: Asynchronously crawls specified documentation websites, extracts content, and processes it into structured chunks.
- Intelligent Chunking: Splits text into manageable chunks, intelligently respecting code blocks, paragraphs, and sentences to maintain context.
- LLM-powered Summarization & Title Extraction: Uses a local Ollama model to generate concise summaries and descriptive titles for each document chunk.
- Vector Embeddings: Generates embeddings for each chunk using a local Ollama embedding model, enabling semantic search.
- Supabase Integration: Stores processed document chunks, along with their metadata and embeddings, in a Supabase database for efficient retrieval.
- Agentic RAG Pipeline:
    - Tool Usage: The main agent (Documentation AI Expert) is equipped with tools to retrieve relevant documentation, list available documentation pages, and fetch full page content from Supabase.
    - Dynamic Prompting: Adapts its system prompt based on previous attempts, encouraging the agent to use all required tools in a specified order.
    - Self-Correction/Validation: Employs a separate validation agent (also a local LLM) to check if the generated answer accurately addresses the original query, leading to iterative refinement.
    - Custom Logging: Provides enhanced logging with colored output and emojis for better debuggability.

# 📁 Project Structure
```
rag_project/
├── .env                  # Environment variables (Supabase, Ollama, Logging, App Config)
├── requirements.txt      # Python dependencies
└── src/
    ├── __init__.py       # Marks 'src' as a Python package
    ├── main.py           # Core RAG agent logic, tool definitions, and main execution loop
    └── crawl.py          # Web crawling, data processing, and Supabase ingestion script
``` 

# ⚙️ Setup
Follow these steps to get the project up and running on your local machine.

- Prerequisites
    - Python 3.9+
    - Docker (for running Supabase and Ollama locally, or ensuring they are accessible)
    - Ollama: Make sure Ollama is running and you have the necessary models pulled.
        - Download Ollama from ollama.com.
        - Pull the required models:
        - ollama pull qwen2.5:14b         # For summarization and title extraction (or your chosen OLLAMA_MODEL_NAME)
        - ollama pull deepseek-coder:14b  # For validation (or your chosen validation model)
        - ollama pull nomic-embed-text    # For embeddings
    - Supabase: You can run Supabase locally using Docker.
        - You can also use a hosted Supabase


## Code Setup
1. Clone the Repository 
First, set up the recommended project structure if you haven't already:
```
git clone https://
```


2. Set up Environment Variables

Create a .env file in the root directory (rag_project/) and populate it with your configuration:

    - Supabase Configuration
        - SUPABASE_URL
        - SUPABASE_SERVICE_KEY
    - Ollama Configuration
        - OLLAMA_BASE_URL=
        - OLLAMA_MODEL_NAME=qwen2.5:14b # Or llama3.2:3b, llama3.1:latest, etc.
    - Embedding Configuration
        - EMBEDDING_API_URL=
        - EMBEDDING_MODEL=nomic-embed-text
    - Logging Configuration
        - LOG_LEVEL=INFO # DEBUG, INFO, WARNING, ERROR, CRITICAL
    - Application Configuration
        - MATCH_COUNT=5 # Number of relevant documents to retrieve

3. Install Dependencies

Create a requirements.txt file in the root directory (rag_project/) with the following content:

    - aiohttp
    - asyncio
    - httpx
    - pydantic_ai
    - supabase
    - python-dotenv
    - crawl4ai

**It's highly recommended to use a Python virtual environment.**

```
python -m venv venv 
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

4. Supabase Database Setup 

You need a site_pages table and a match_site_pages function in your Supabase database.

site_pages Table Schema
Create a table named site_pages with the following schema:
```
CREATE TABLE public.site_pages (
    id bigint primary key generated always as identity,
    url text,
    chunk_number int,
    title text,
    summary text,
    content text,
    metadata jsonb,
    embedding vector(1536) -- Ensure this matches your embedding model's dimension (nomic-embed-text uses 1536)
);
```
- **Note: Enable the pg_vector extension if not already enabled**
```
-- CREATE EXTENSION IF NOT EXISTS vector;
```
- **match_site_pages** PostgreSQL Function: This function enables semantic search over your embeddings.

```
create or replace function match_site_pages(
    query_embedding vector(1536),
    match_count int DEFAULT null,
    filter jsonb DEFAULT '{}'
)
returns table (
    id bigint,
    url text,
    chunk_number int,
    title text,
    summary text,
    content text,
    metadata jsonb,
    embedding vector(1536)
)
language plpgsql
as $$
#variable_conflict use_column
begin
    return query
    select
        id,
        url,
        chunk_number,
        title,
        summary,
        content,
        metadata,
        embedding
    from public.site_pages
    where metadata @> filter
    order by public.site_pages.embedding <=> query_embedding
    limit match_count;
end;
$$;
```

- **Row Level Security (RLS) Policy for site_pages:**
You might want to create an RLS policy to allow anon role to SELECT data, but for service_role and authenticated users to INSERT. For local development, you might disable RLS or set a permissive policy.

- Allow anonymous users to read public site_pages
```
create policy "Enable read access for all users"
on public.site_pages
for select
using (true);
```
- Allow authenticated users to insert data (*or service_role, if using via backend*)
```
create policy "Enable insert for authenticated users"
on public.site_pages
for insert
with check (auth.role() = 'authenticated' or auth.role() = 'service_role');
```

# 🚀 Usage
1. Run the Crawler (Data Ingestion): First, populate your Supabase database by running the crawler. This will fetch documentation from https://xyz.com/sitemap.xml, process it, and store it.
```
python src/crawl.py
```
This process might take some time depending on the number of URLs and your Ollama setup.

2. Run the RAG Agent: Once the database is populated, you can run the main RAG agent to answer questions.
```
python src/main.py
```
#### The main.py script currently has a hardcoded question (original_question = "get me the Weather Agent Example"). You can modify this variable to ask different questions. Observe the logging output to see the agent's attempts, tool usage, and validation process.


# 🤝 Extending with Model Context Protocol (MCP) Servers
This project provides a strong foundation for an agentic RAG. To further enhance its capabilities and achieve even better results, consider integrating Model Context Protocol (MCP) servers.

## MCP servers can:

- Standardize External Interactions: Abstract away direct API calls (like to Supabase or Ollama) behind a consistent MCP interface.

- Add New Tools: Easily integrate new functionalities (e.g., advanced web search, specific APIs, specialized data processing) as MCP tools that your agent can discover and use.

- Improve Agentic Behavior: By providing a more robust and modular tool-calling mechanism, MCP can help the agent make more reliable decisions about when and how to use external resources.

- Explore the Awesome MCP Servers repository for a curated list of available MCP servers that you can integrate into your project. For example, you could replace direct Supabase calls with a supabase-mcp-server or introduce exa-mcp-server for powerful web search capabilities.

# 📜 License
This project is open-source and available under the MIT License.