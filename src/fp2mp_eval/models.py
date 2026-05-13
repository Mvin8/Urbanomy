from pydantic import BaseModel, Field

class Dimension(BaseModel):
    score : int = Field(ge=1, le=5)
    evidence : str = Field(description="Specific evidence from the solution trace supporting the score.")
    commentary : str = Field(description="What is missing or weak.")

class Evaluation(BaseModel):
    framing : Dimension = Field(description='Quality of problem framing: identification of key elements, goals, assumptions.')
    decomposition : Dimension = Field(description='How well the problem is broken into subproblems and structured.')
    diversity : Dimension = Field(description='Number and quality of alternative solutions considered.')
    coherence : Dimension = Field(description='Logical consistency and step-by-step reasoning clarity.')
    justification : Dimension = Field(description="Strength of argumentation and trade-off analysis.")
    uncertainty_handling : Dimension = Field(description="Handling of missing information, assumptions, ambiguity.")
    knowledge_integration : Dimension = Field(description="Use and transfer of relevant domain knowledge.")
    metacognition : Dimension = Field(description="Reflection on reasoning process and limitations.")

__all__=[
    "Dimension",
    "Evaluation"
]