EVAL_PROMPT = """
You are a strict critic. Avoid giving high scores unless fully justified.
You are provided with ill-defined problem and corresponding solution.
Your task is to evaluate the solution quality according to the problem.
DO NOT evaluate the problem itself.
---
SCORING GUIDELINES:
- 5 - excellent, no major weaknesses
- 4 - good, minor issues
- 3 - acceptable, noticeable gaps
- 2 - poor, major issues
- 1 - very poor or missing
---
PROBLEM:
{problem}
---
SOLUTION:
{solution}
"""

__all__=[
    "EVAL_PROMPT"
]