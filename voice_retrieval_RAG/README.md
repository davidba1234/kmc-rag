# Voice-Optimized RAG Retrieval System

This module implements a high-performance retrieval system for RAG, optimized for voice interactions using Supabase Hybrid Search and FlashRank re-ranking.

## Prerequisites

- Python 3.8+
- Supabase project with `vector` extension enabled.
- OpenAI API Key.

## Setup

1.  **Set up Virtual Environment** (Recommended):
    -   It is highly recommended to use a separate virtual environment to avoid conflicts.
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    
    # Mac/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables**:
    - Copy `.env.example` to `.env`:
        ```bash
        cp .env.example .env
        ```
    - Fill in your `SUPABASE_URL`, `SUPABASE_KEY`, and `OPENAI_API_KEY`.

3.  **Database Setup**:
    - Run the SQL commands in `setup.sql` in your Supabase SQL Editor.
    - **Note**: Ensure your table is named `documents` and the vector column is `embedding`. If not, edit `setup.sql` and `retrieval.py` accordingly.

## Usage

Run the retrieval script:

```bash
python retrieval.py
```

You can modify the `voice_query` variable in the `if __name__ == "__main__":` block to test different queries.

## Architecture

1.  **Input**: User voice query (converted to text).
2.  **Embedding**: Generate embedding for the query using OpenAI `text-embedding-3-small`.
3.  **Hybrid Search**: Supabase searches for candidates using both Vector Search (HNSW index) and Full Text Search.
4.  **Re-ranking**: FlashRank re-ranks the top 25 candidates locally to find the most relevant top 3.
5.  **Output**: Top 3 text chunks for LLM context.
