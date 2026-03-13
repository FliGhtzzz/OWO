"""
LLM integration - Ollama API calls for keyword extraction, image description, and query expansion.
All outputs are in English for consistent indexing.
"""
import httpx
import base64
import json
import re
import logging
import os
import asyncio
from typing import Optional, Union, Any

# Setup AI Logger
ai_log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai.log")
ai_logger = logging.getLogger("ai_engine")
ai_logger.setLevel(logging.INFO)
# Clear existing handlers
if ai_logger.handlers:
    ai_logger.handlers.clear()
handler = logging.FileHandler(ai_log_file, encoding='utf-8')
handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
ai_logger.addHandler(handler)

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
TIMEOUT = 300.0  # seconds - local model can be slow

# Simple cache for the model name to avoid constant DB reads
_cached_model = None
_cache_time = 0
CACHE_TTL = 30 # seconds

def get_model_name() -> str:
    """Get the current selected model name from database with caching."""
    global _cached_model, _cache_time
    import time
    from database import get_setting

    now = time.time()
    if _cached_model is None or (now - _cache_time) > CACHE_TTL:
        _cached_model = get_setting("llm_model", "gemma3:4b")
        _cache_time = now
    return _cached_model


def _clear_llm_cache():
    """Clear the cached model name to force a refresh from the database."""
    global _cached_model
    _cached_model = None


async def _chat(prompt: str, image_path: Optional[str] = None,
                num_predict: Optional[int] = None,
                timeout: Optional[float] = None) -> str:
    """Send a chat request to Ollama using streaming to respect asyncio cancellation."""
    model = get_model_name().strip()
    messages = [{"role": "user", "content": prompt}]

    # Use precise per-operation timeouts so asyncio.wait_for can actually cancel
    read_timeout = timeout if timeout is not None else TIMEOUT
    http_timeout = httpx.Timeout(
        connect=5.0,       # Quick fail if Ollama isn't running
        read=read_timeout, # Max time between receiving chunks (not total time)
        write=10.0,
        pool=5.0,
    )

    if image_path:
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")
        messages[0]["images"] = [img_data]

    options: dict[str, object] = {
        "temperature": 0.1,
    }
    if num_predict is not None:
        options["num_predict"] = num_predict

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,   # Stream so we can cancel mid-generation
        "options": options,
    }

    async with httpx.AsyncClient(timeout=http_timeout) as client:
        try:
            import time
            start_time = time.time()
            chunks = []
            async with client.stream("POST", f"{OLLAMA_BASE_URL}/api/chat", json=payload) as response:
                if response.status_code == 404:
                    raise Exception(f"Model '{model}' not found in Ollama. Please download it or select another model.")
                response.raise_for_status()
                async for line in response.aiter_lines():
                    # Stop generation if it has taken longer than the overall timeout
                    if time.time() - start_time > read_timeout:
                        ai_logger.warning(f"LLM hard timeout ({read_timeout}s) reached. Truncating output.")
                        break

                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                        delta = chunk.get("message", {}).get("content", "")
                        if delta:
                            chunks.append(delta)
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue

            content = "".join(chunks)
            ai_logger.info(f"Model '{model}' responded. Content length: {len(content)}")
            ai_logger.debug(f"Raw Output: {content}")
            return content
        except httpx.ConnectError:
            raise Exception("Cannot connect to Ollama. Is it running?")
        except Exception as e:
            raise e



def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences like ```json ... ``` that Qwen/other models add."""
    # Remove ```json ... ``` or ``` ... ``` blocks, keeping only the inner content
    text = re.sub(r'^```(?:json|JSON)?\s*', '', text.strip())
    text = re.sub(r'```\s*$', '', text.strip())
    # Also handle inline fences in the middle
    text = re.sub(r'```(?:json|JSON)?([\s\S]*?)```', r'\1', text)
    return text.strip()


def _clean_keyword_list(keywords: Union[list[str], str, Any]) -> list[str]:
    """Uniformly clean a list of keywords, removing code and syntax."""
    if isinstance(keywords, str):
        keywords = [k.strip() for k in re.split(r'[;,\n]', keywords) if k.strip()]
    elif not isinstance(keywords, list):
        return []
        
    clean_list = []
    seen = set()
    for kw in keywords:
        if not kw: continue
        kw = str(kw).lower() # convert to lowercase for deduplication
        # Remove all common code/syntax symbols
        kw_clean = re.sub(r'[{}()[\];<>#=*&%@!_.,"\'+-/]', ' ', kw)
        # Collapse spaces
        kw_clean = ' '.join(kw_clean.split())
        # Filter if it still looks like code or is too short.
        # Allow single-character keywords if they are non-ASCII (e.g. Chinese characters).
        is_short_ascii = len(kw_clean) <= 1 and all(ord(c) < 128 for c in kw_clean)
        if kw_clean and not _is_likely_code(kw_clean) and not is_short_ascii and len(kw_clean) < 40:
            if kw_clean not in seen:
                seen.add(kw_clean)
                clean_list.append(kw_clean)
    return clean_list[:50] # Strictly enforce max 50 keywords in the code as well


def _truncate_keywords(keywords: str, max_chars: int = 1500) -> str:
    """Truncate a space-separated keyword string to max_chars at word boundaries."""
    if len(keywords) <= max_chars:
        return keywords
    truncated = keywords[:max_chars]
    # Step back to the last space to avoid cutting a word in half
    last_space = truncated.rfind(' ')
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated


def _is_likely_code(line: str) -> bool:
    """Detect if a line of text is likely source code rather than keywords."""
    # Common code syntax markers
    code_markers = [
        '#include', 'using namespace', 'int ', 'void ', 'public:', 'private:',
        '{', '}', ';', '(', ')', '[', ']', 'cout <<', 'print(', 'def ', 'class ',
        'import ', 'from ', 'std::', 'return ', 'if (', 'while ('
    ]
    line_low = line.lower()
    
    # Check for markers
    marker_count = sum(1 for m in code_markers if m in line_low)
    
    # High concentration of punctuation also suggests code
    punc_count = sum(1 for c in line if c in '(){}[];=<>#&*')
    
    return marker_count >= 1 or punc_count > 5


def _cleanup_expansion_result(response: str, user_query: str) -> str:
    """Clean up LLM expansion response, stripping code and explanations."""
    if not response:
        return user_query
        
    # Remove quotes
    keywords = response.strip().strip('"').strip("'")
    
    # Split by lines and filter
    lines = keywords.split("\n")
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Ignore lines that look like code
        if _is_likely_code(line):
            continue
            
        # Ignore common prefixes
        line = re.sub(r'^(keywords:|answer:|result:|search terms:)\s*', '', line, flags=re.IGNORECASE)
        
        # If the line is mostly a single long sentence, it might be an explanation
        if len(line.split()) > 20 and '.' in line:
            continue
            
        cleaned_lines.append(line)

    if not cleaned_lines:
        return user_query
        
    # Usually the expansion is the line with many words but no punctuation
    # Sort by "keyword density" vs "punctuation density"
    def score_line(l):
        words = l.split()
        if not words: return -1
        punc = sum(1 for c in l if c in '.,!?;:"()[]{}')
        return len(words) - (punc * 2)

    best_line = max(cleaned_lines, key=score_line)
    
    # Final cleanup of remaining syntax - remove EVERYTHING that isn't a word or space
    # Replace common symbols with space, then collapse multiple spaces
    best_line = re.sub(r'[{}()[\];<>#=*&%@!_.,"\'+-/]', ' ', best_line)
    # Remove extra spaces
    best_line = ' '.join(best_line.split())
    
    # Simple deduplication of words in the expanded query
    words = best_line.split()
    seen = set()
    deduped = []
    for w in words:
        wl = w.lower()
        if wl not in seen:
            seen.add(wl)
            deduped.append(w)
            
    return ' '.join(deduped[:30]) # keep max 30 words for search expansion


def _parse_json_response(text: str) -> dict:
    """Try to extract JSON from LLM response with high resilience.
    Handles Qwen-style markdown fences, trailing commas, and other quirks.
    """
    if not text:
        return {"summary": "", "keywords": []}

    # Step 0: Strip markdown code fences (Qwen, Mistral etc. love adding these)
    text = _strip_markdown_fences(text).strip()

    data = None

    # Step 1: Try first { ... last } extraction
    first_brace = text.find('{')
    last_brace = text.rfind('}')

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_str = text[first_brace:last_brace+1]
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Fix trailing commas, then retry
            json_str_fixed = re.sub(r',\s*([\]}])', r'\1', json_str)
            try:
                data = json.loads(json_str_fixed)
            except json.JSONDecodeError:
                pass

    if data and isinstance(data, dict):
        # Case-insensitive key lookup
        data_low = {k.lower(): v for k, v in data.items()}

        # Keywords extraction
        keywords = []
        for key in ["keywords", "tags", "keyword_list", "entities", "labels"]:
            if key in data_low:
                val = data_low[key]
                if isinstance(val, str):
                    keywords = [k.strip() for k in re.split(r'[;,\n]', val) if k.strip()]
                elif isinstance(val, list):
                    keywords = [str(k).strip() for k in val if k]
                break

        # Summary extraction
        summary = ""
        for key in ["summary", "description", "abstract", "content"]:
            if key in data_low:
                summary = str(data_low[key]).strip()
                break

        if summary and keywords:
            ai_logger.info(f"JSON parsed OK: {len(keywords)} keywords.")
            # Truncate summary to 600 chars at a sentence boundary
            if len(summary) > 600:
                cut = summary[:600]
                dot = cut.rfind(".")
                summary = (cut[:dot + 1] if dot > 200 else cut).strip()
            return {"summary": summary, "keywords": _clean_keyword_list(keywords)}

        if keywords:  # have keywords but no summary
            summary = text[:300] + "..." if len(text) > 300 else text
            ai_logger.info(f"JSON partial: {len(keywords)} keywords, no summary.")
            return {"summary": summary, "keywords": _clean_keyword_list(keywords)}

    # Step 2: Regex fallback — try to extract keywords array directly
    # Handles cases like:  "keywords": ["a", "b", "c"]
    kw_array_match = re.search(
        r'["\']?keywords["\']?\s*:\s*\[([^\]]+)\]', text, re.IGNORECASE | re.DOTALL
    )
    if kw_array_match:
        raw_items = kw_array_match.group(1)
        # Extract quoted strings or bare words
        keywords = re.findall(r'["\']([^"\']+)["\']|([^,\[\]\n"\'\.]+)', raw_items)
        keywords = [a or b for a, b in keywords]
        keywords = [k.strip() for k in keywords if k.strip()]
        if keywords:
            ai_logger.info(f"Regex array fallback: {len(keywords)} keywords.")
            return {"summary": "", "keywords": _clean_keyword_list(keywords)}

    # Step 3: Line-by-line parsing for bullet-point style outputs
    keywords = []
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith(("- ", "* ", "• ")) and len(line) > 2:
            keywords.append(line[2:].strip())
        elif ":" in line and any(k in line.lower() for k in ["keywords", "tags", "labels"]):
            parts = line.split(":", 1)[1]
            keywords.extend([k.strip() for k in re.split(r'[;,\n]', parts) if k.strip()])

    clean_text = re.sub(r'```.*?```', '', text, flags=re.DOTALL).strip()
    result = {"summary": clean_text if clean_text else text.strip(), "keywords": _clean_keyword_list(list(set(keywords)))}
    ai_logger.info(f"Fallback parse: {len(result['keywords'])} keywords found.")
    return result


# Custom exception for LLM failures - ensures callers never get polluted data
class LLMFailedError(Exception):
    """Raised when LLM fails to generate content after all retries."""
    pass


async def extract_keywords(text: str, file_name: str) -> dict:
    """
    Extract keywords and summary from text content.
    Returns: {"summary": str, "keywords": [str]}
    Raises LLMFailedError if LLM fails after 2 attempts - caller must handle.
    """
    prompt = f"""You are a precise search indexing engine. Analyze the file below and extract high-quality metadata for search.
File name: {file_name}

CONTENT:
{text[:3000]}

RULES:
- Respond ONLY with a valid JSON object. No markdown, no code fences, no extra text.
- All output must be in English. Translate non-English concepts but preserve original proper nouns as extra keywords.
- Summary: 2-4 sentences. Be SPECIFIC: mention exact algorithms, methods, topics, names, places. No vague filler.
- Keywords: Select the most critical keywords, MAXIMUM 30 keywords. Each keyword MUST be concise (under 30 characters). MAXIMUM 30 KEYWORDS TOTAL. DO NOT GENERATE MORE.
  - INCLUDE: algorithm names, data structures, proper nouns, technical terms, domain-specific concepts, file-specific topics
  - FOR ANIMALS, PLANTS, OR SPECIFIC OBJECTS: Always include the general category/hypernym (e.g., if you find "Dachshund", also include "dog"; if "Oak", include "tree").
  - EXCLUDE generic filler words: "code", "data", "file", "list", "function", "example", "implementation", "snippet", "programming", "computer", "method", "value", "result", "output", "input", "variable"
  - EXTREMELY IMPORTANT: Every keyword must be UNIQUE. Avoid exact duplicates, but include both specific terms and their general categories (hypernyms).
  - Multi-word specific terms MUST stay together: "binary search", "machine learning", "dynamic programming"
  - Proper nouns in original language are OK if they add value: "台北101", "東京"

OUTPUT FORMAT (JSON ONLY):
{{"summary": "Specific description of what this file is actually about", "keywords": ["specific-term1", "specific-term2", "general-category"]}}

EXTREMELY IMPORTANT: STOP GENERATING immediately after the closing brace "}}". Do not append any trailing text."""

    ai_logger.info(f"Extracting keywords for {file_name}...")
    for attempt in range(2):
        try:
            # 20s timeout per attempt
            response = await _chat(prompt, num_predict=300, timeout=20.0)
            result = _parse_json_response(response)
            
            # Basic validation
            if not result.get("summary") and not result.get("keywords"):
                raise Exception("LLM returned empty or unparseable result")
                
            if "summary" not in result:
                result["summary"] = ""
            if "keywords" not in result:
                result["keywords"] = []
            return result
        except Exception as e:
            ai_logger.warning(f"[Attempt {attempt+1}/2] Error for {file_name}: {e}")
            if attempt == 0:
                ai_logger.info(f"Retrying {file_name}...")
                continue
            # Both attempts failed — raise to prevent database pollution
            raise LLMFailedError(f"extract_keywords failed for '{file_name}' after 2 attempts") from e


async def describe_image(image_path: str, file_name: str) -> dict:
    """
    Describe an image using vision model.
    Returns: {"summary": str, "keywords": [str]}
    Raises LLMFailedError if LLM fails after 2 attempts - caller must handle.
    """
    prompt = f"""You are a precise visual search indexing engine. Analyze this image and extract high-quality metadata.
File name: {file_name}

ANALYZE THESE ASPECTS:
1. VISIBLE TEXT & OCR: Transcribe ALL text visible in the image (signs, labels, screens, watermarks, code, formulas). Only if actually present.
2. OBJECTS & SCENE: List ALL concrete objects, materials, colors, setting (indoor/outdoor), lighting, and composition.
3. PEOPLE & ACTIONS: If people are present, describe their actions, attire, posture. Do NOT invent.
4. PROPER NOUNS (HIGH PRIORITY):
   - Place names, landmarks, brand names, company logos, product names
   - Keep original language for proper nouns (e.g., "台北101", "東京タワー")

RULES:
- Respond ONLY with a JSON object. No markdown fences, no extra text.
- Summary: 2-4 sentences. Be SPECIFIC. Mention the most distinctive elements.
- Keywords: Select the most critical keywords, MAXIMUM 30 keywords. Each keyword MUST be concise (under 20 characters). 
  - INCLUDE: object names, scene descriptors, transcribed text, proper nouns, colors, actions, mood
  - FOR ANIMALS, PLANTS, OR SPECIFIC OBJECTS: Always include the general category/hypernym (e.g., if you find "Dachshund", also include "dog").
  - EXCLUDE vague words: "image", "photo", "picture", "file", "background", "scene" (unless scene type is specific like "beach scene")
  - EXTREMELY IMPORTANT: Each item MUST be unique! Avoid exact duplicates, but include both specific terms and their general categories (hypernyms).
  - Multi-word terms stay together: "Eiffel Tower", "golden retriever", "brick wall"
- DO NOT hallucinate. Only describe what is actually visible.

OUTPUT FORMAT (JSON ONLY):
{{"summary": "Specific description of the image content", "keywords": ["specific-term1", "specific-term2", "general-category"]}}

EXTREMELY IMPORTANT: STOP GENERATING immediately after the closing brace "}}". Do not append any trailing text."""

    ai_logger.info(f"Describing image {file_name}...")
    for attempt in range(2):
        try:
            # 25s timeout per attempt for vision
            response = await _chat(prompt, image_path=image_path, num_predict=600, timeout=25.0)
            result = _parse_json_response(response)
            
            # Basic validation
            if not result.get("summary") and not result.get("keywords"):
                raise Exception("LLM returned empty or unparseable result")

            if "summary" not in result:
                result["summary"] = ""
            if "keywords" not in result:
                result["keywords"] = []
            return result
        except Exception as e:
            ai_logger.warning(f"[Attempt {attempt+1}/2] Error describing image {file_name}: {e}")
            if attempt == 0:
                ai_logger.info(f"Retrying {file_name}...")
                continue
            # Both attempts failed — raise to prevent database pollution
            raise LLMFailedError(f"describe_image failed for '{file_name}' after 2 attempts") from e


async def expand_query(user_query: str) -> str:
    """
    Expand a natural language query into comprehensive English search keywords.
    Returns a space-separated string of keywords for Elasticsearch search.
    """
    user_query = user_query.strip()

    prompt = f"""You are a file search assistant. The user typed a query and you must extract precise English search keywords to find matching files on their computer.

USER QUERY: "{user_query}"

STEPS:
1. Understand the user's TRUE INTENT.
2. If the query is not in English, TRANSLATE IT to its exact English equivalent first.
3. Include BOTH the common short name AND the full scientific/official name for any animal, place, or concept.
4. Add 3-5 closely related synonyms or specific sub-topics.

EXAMPLES:
- "二分搜" → binary search bisect sorted array divide conquer
- "海邊照片" → beach ocean sea coast shoreline waves sand seaside
- "Python 遞迴" → recursion recursive function Python base case call stack
- "菲律賓" → Philippines Filipino Manila island southeast asia tropical
- "河馬" → hippo hippopotamus water animal Africa river mammal zoo
- "hippo" → hippo hippopotamus river mammal Africa water zoo wildlife
- "cat" → cat kitten feline tabby cat domestic animal pet
- "大象" → elephant mammal Africa Asia trunk ivory wildlife zoo safari

CRITICAL RULES:
- ENTIRE OUTPUT MUST BE IN ENGLISH. No Chinese characters allowed in output.
- The DIRECT ENGLISH TRANSLATION of the user's core concept MUST be the very first word(s).
- Output ONLY a single line of space-separated English keywords. No explanations, no commas.
- Select the most critical keywords, MAXIMUM 10 keywords. QUALITY over quantity.
- For animals/places, ALWAYS include BOTH the specific name AND the general category (e.g. "Dachshund" AND "dog", "hippo" AND "mammal").
- ABSOLUTELY NO REPETITIVE LOOPS. Each word MUST be unique!
- Prioritize SPECIFIC terms over generic ones. Avoid: "file", "data", "picture", "photo", "image"."""

    try:
        response = await _chat(prompt, num_predict=300)
        # Clean up the response
        if not response:
            return user_query

        # Remove quotes if present
        keywords = response.strip().strip('"').strip("'")

        # Remove any lines that look like explanations
        lines = keywords.split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line and len(line.split()) > 1:
                keywords = line
                break

        # Remove common prefixes LLMs might add
        keywords = re.sub(r'^(keywords:|answer:|result:|here are the keywords:)\s*', '', keywords, flags=re.IGNORECASE)

        # Truncate and deduplicate words to prevent infinite repetition loops
        words = keywords.split()
        seen = set()
        deduped = []
        for w in words:
            wl = w.lower()
            if wl not in seen:
                seen.add(wl)
                deduped.append(w)
                
        keywords = ' '.join(deduped[:30])
        
        # Truncate to 1500 chars at word boundary to avoid overly long search queries
        keywords = _truncate_keywords(keywords, max_chars=1500)

        return str(keywords)
    except Exception as e:
        print(f"[LLM] Error expanding query: {e}")
        return user_query


async def check_ollama_status() -> dict:
    """Check if Ollama is running and model is available."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Check if Ollama is running
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
            models = resp.json().get("models", [])
            model_names = [m["name"].strip() for m in models]
            
            current_model = get_model_name().strip()
            # Loose match: check if the selected model name (before version) is in the available models
            model_base = current_model.split(":")[0]
            has_model = any(model_base in name or current_model in name for name in model_names)

            return {
                "ollama_running": True,
                "model_available": has_model,
                "available_models": model_names,
                "selected_model": current_model,
            }
    except Exception as e:
        return {
            "ollama_running": False,
            "model_available": False,
            "error": str(e),
            "selected_model": get_model_name(),
        }


async def expand_query_with_file(user_query: str, file_content: Optional[str] = None,
                                  image_path: Optional[str] = None) -> str:
    """
    Expand a search query using both text and an uploaded file.
    The LLM analyzes the file content/image + user query together to
    generate comprehensive search keywords.
    """
    context_parts = []

    if user_query:
        context_parts.append(f"USER TEXT QUERY: {user_query}")

    if file_content:
        context_parts.append(f"UPLOADED FILE CONTENT:\n{file_content[:3000]}")

    context = "\n\n".join(context_parts)

    prompt = f"""You are a precise file search assistant. The user uploaded a file and/or typed a query to find related files.

CONTEXT:
{context}

TASK:
1. Understand the user's intent: what content/topic/file type are they searching for?
2. Extract the core subject matter from the provided file content or image.
3. Generate specific English search keywords that would match related files.

EXAMPLES OF GOOD KEYWORDS:
- For a Python binary search file: bisect binary-search sorted-array O(log-n) divide-conquer half-interval
- For a beach photo: beach ocean coast waves sand sunset tropical shoreline seascape
- For meeting notes: meeting minutes agenda action-items discussion decisions

CRITICAL RULES:
- Output ONLY a single line of space-separated English keywords.
- Select the most critical keywords, MAXIMUM 50 keywords. Each keyword MUST be concise (under 30 characters). Prioritize SPECIFIC, SEARCHABLE terms.
- DO NOT repeat word variants of the same concept.
- DO NOT include raw code syntax (brackets, semicolons, operators). Extract concepts only.
- Avoid generic words: "code", "file", "data", "implementation", "example"."""

    try:
        response = await _chat(prompt, image_path=image_path, num_predict=300)
        if not response:
            return user_query or ""

        # Clean up
        keywords = response.strip().strip('"').strip("'")
        lines = keywords.split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line and len(line.split()) > 1:
                keywords = line
                break

        keywords = re.sub(r'^(keywords:|answer:|result:|here are the keywords:)\s*', '', keywords, flags=re.IGNORECASE)

        # Truncate to 1500 chars at word boundary
        keywords = _truncate_keywords(keywords, max_chars=1500)

        return keywords
    except Exception as e:
        print(f"[LLM] Error expanding query with file: {e}")
        return user_query or ""
