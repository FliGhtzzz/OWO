import re

# Mocking the functions from llm.py for testing
def _is_likely_code(line: str) -> bool:
    code_markers = [
        '#include', 'using namespace', 'int ', 'void ', 'public:', 'private:',
        '{', '}', ';', '(', ')', '[', ']', 'cout <<', 'print(', 'def ', 'class ',
        'import ', 'from ', 'std::', 'return ', 'if (', 'while ('
    ]
    line_low = line.lower()
    marker_count = sum(1 for m in code_markers if m in line_low)
    punc_count = sum(1 for c in line if c in '(){}[];=<>#&*')
    return marker_count >= 1 or punc_count > 5

def _cleanup_expansion_result(response: str, user_query: str) -> str:
    if not response:
        return user_query
    keywords = response.strip().strip('"').strip("'")
    lines = keywords.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line: continue
        if _is_likely_code(line): continue
        line = re.sub(r'^(keywords:|answer:|result:|search terms:)\s*', '', line, flags=re.IGNORECASE)
        if len(line.split()) > 20 and '.' in line: continue
        cleaned_lines.append(line)
    if not cleaned_lines: return user_query
    def score_line(l):
        words = l.split()
        if not words: return -1
        punc = sum(1 for c in l if c in '.,!?;:"()[]{}')
        return len(words) - (punc * 2)
    best_line = max(cleaned_lines, key=score_line)
    best_line = re.sub(r'[{}()[\];<>#=*&%@!]', ' ', best_line)
    return ' '.join(best_line.split())

# Test cases
tests = [
    {
        "name": "Code at the end",
        "response": "Here are keywords:\nbinary search algorithm array sorting\n#include <iostream>\nint n = 8;",
        "expected": "binary search algorithm array sorting"
    },
    {
        "name": "Hallucinated code in middle",
        "response": "Keywords: binary search\nprint(__import__('bitsect')...)\nbisect left module",
        "expected": "bisect left module" # or binary search depending on score
    },
    {
        "name": "No code, just keywords",
        "response": "binary search algorithm tree",
        "expected": "binary search algorithm tree"
    }
]

for t in tests:
    result = _cleanup_expansion_result(t['response'], "fallback")
    print(f"Test: {t['name']}")
    print(f"Result: '{result}'")
    # Simple check
    if any(c in result for c in '{};#'):
        print("FAILED: Result contains code-like characters")
    print("-" * 20)
