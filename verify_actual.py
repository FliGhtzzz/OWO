import asyncio
import re
from llm import _cleanup_expansion_result, _is_likely_code

# Test with the ACTUAL user example from screenshot
test_input = """bisect algorithm binary search list manipulation input parsing Python code efficiency data structures indexing search operations
print(__import__("bitsect").bisect(*[int(input().split()[1]),list(map(int, input().split()))][::-1]))"""

print("--- Testing _is_likely_code ---")
lines = test_input.split("\n")
for i, line in enumerate(lines):
    is_code = _is_likely_code(line)
    print(f"Line {i+1} is_code: {is_code}")
    # Show markers found
    markers = [m for m in ['#include', 'print(', 'import ', '(', ')', '[', ']'] if m in line.lower()]
    print(f"  Markers: {markers}")

print("\n--- Testing _cleanup_expansion_result ---")
result = _cleanup_expansion_result(test_input, "FALLBACK")
print(f"Result:\n'{result}'")

if "print(" in result:
    print("\nFAILED: Code leaked into result!")
else:
    print("\nSUCCESS: Code was filtered out.")
