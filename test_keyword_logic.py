
import asyncio
import json
import re
from typing import Union, Any

# Mocking the cleaning functions from llm.py to verify they don't over-deduplicate
def _is_likely_code(line: str) -> bool:
    code_markers = ['#include', 'void ', '{', '}', ';', '(', ')', 'print(']
    line_low = line.lower()
    marker_count = sum(1 for m in code_markers if m in line_low)
    punc_count = sum(1 for c in line if c in '(){}[];=<>#&*')
    return marker_count >= 1 or punc_count > 5

def _clean_keyword_list(keywords: Union[list[str], str, Any]) -> list[str]:
    if isinstance(keywords, str):
        keywords = [k.strip() for k in re.split(r'[;,\n]', keywords) if k.strip()]
    elif not isinstance(keywords, list):
        return []
        
    clean_list = []
    seen = set()
    for kw in keywords:
        if not kw: continue
        kw = str(kw).lower()
        kw_clean = re.sub(r'[{}()[\];<>#=*&%@!_.,"\'+-/]', ' ', kw)
        kw_clean = ' '.join(kw_clean.split())
        
        # Allow single-character keywords if they are non-ASCII (e.g. Chinese characters).
        is_short_ascii = len(kw_clean) <= 1 and all(ord(c) < 128 for c in kw_clean)
        if kw_clean and not _is_likely_code(kw_clean) and not is_short_ascii and len(kw_clean) < 40:
            if kw_clean not in seen:
                seen.add(kw_clean)
                clean_list.append(kw_clean)
    return clean_list[:50]

def test_categorical_hierarchy():
    print("Testing categorical hierarchy preservation...")
    
    # Case 1: Specific breed and general category
    input_keywords = ["Dachshund", "dog", "臘腸狗", "狗", "mammal"]
    cleaned = _clean_keyword_list(input_keywords)
    print(f"Input: {input_keywords}")
    print(f"Cleaned: {cleaned}")
    
    assert "dachshund" in cleaned
    assert "dog" in cleaned
    assert "臘腸狗" in cleaned
    assert "狗" in cleaned
    assert "mammal" in cleaned
    print("PASSED: Mixed hierarchy preserved.\n")

    # Case 2: Redundant but distinct terms
    input_keywords = ["Golden Retriever", "Retriever", "Dog", "Pet"]
    cleaned = _clean_keyword_list(input_keywords)
    print(f"Input: {input_keywords}")
    print(f"Cleaned: {cleaned}")
    
    assert "golden retriever" in cleaned
    assert "retriever" in cleaned
    assert "dog" in cleaned
    print("PASSED: Redundant but distinct terms preserved.\n")

    # Case 3: Actual duplicates (should still be removed)
    input_keywords = ["dog", "Dog", "DOG", "hound"]
    cleaned = _clean_keyword_list(input_keywords)
    print(f"Input: {input_keywords}")
    print(f"Cleaned: {cleaned}")
    
    assert len(cleaned) == 2
    assert "dog" in cleaned
    assert "hound" in cleaned
    print("PASSED: Actual duplicates removed.\n")

if __name__ == "__main__":
    test_categorical_hierarchy()
    print("All logic tests passed!")
