"""
Searcher - Query expansion + Elasticsearch search.
"""
from llm import expand_query
from database import search as db_search


async def search_files(query: str, limit: int = 20) -> dict:
    """
    Search files using natural language query.
    1. Expand query with LLM (add synonyms, translate to English)
    2. Search Elasticsearch with expanded keywords
    3. Return ranked results
    """
    try:
        if not query:
            print("[Search] Empty query, returning all files")
            expanded_query = ""
            results = db_search(expanded_query, limit=limit)
        else:
            # Step 1: Expand query
            expanded_query = await expand_query(query)
            print(f"[Search] Original: '{query}' → Expanded: '{expanded_query}'")
            # Step 2: Elasticsearch search
            results = db_search(expanded_query, limit=limit)

        # Step 3: Format results
        return {
            "original_query": query,
            "expanded_query": expanded_query,
            "total_results": len(results),
            "results": results,
        }
    except Exception as e:
        print(f"[Search] Error searching for '{query}': {e}")
        # Return empty result instead of crashing
        return {
            "original_query": query,
            "expanded_query": query,  # Fallback
            "total_results": 0,
            "results": [],
            "error": str(e)
        }
