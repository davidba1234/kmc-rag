-- Enable the vector extension if not already enabled
create extension if not exists vector;

-- Create an HNSW index for faster vector search
-- IMPORTANT: Replace 'documents' with your actual table name if different
-- IMPORTANT: Replace 'embedding' with your actual vector column name if different
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx 
ON documents USING hnsw (embedding vector_cosine_ops);

-- Create the hybrid search function
-- This combines Full Text Search (keyword matching) with Vector Search (semantic matching)
create or replace function hybrid_search(
  query_text text,
  query_embedding vector(1536), -- Ensure this matches your embedding model dimensions (1536 for OpenAI text-embedding-3-small)
  match_count int,
  full_text_weight float default 1.0,
  semantic_weight float default 1.0
)
returns setof documents
language sql
as $$
with full_text as (
  select id, row_number() over(order by ts_rank(to_tsvector('english', content), plainto_tsquery('english', query_text)) desc) as rank_ix
  from documents
  where to_tsvector('english', content) @@ plainto_tsquery('english', query_text)
  limit 30
),
semantic as (
  select id, row_number() over(order by embedding <=> query_embedding) as rank_ix
  from documents
  order by embedding <=> query_embedding
  limit 30
)
select documents.*
from documents
join (
    select coalesce(full_text.id, semantic.id) as id,
           coalesce(1.0 / (50 + full_text.rank_ix), 0.0) * full_text_weight +
           coalesce(1.0 / (50 + semantic.rank_ix), 0.0) * semantic_weight as score
    from full_text
    full outer join semantic on full_text.id = semantic.id
    order by score desc
    limit match_count
) as merged on documents.id = merged.id;
$$;
