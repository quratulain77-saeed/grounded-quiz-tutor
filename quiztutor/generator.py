"""Quiz generation from lecture text."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from .grounding import check_quote
from .llm import LLM
from .text import truncate

SYSTEM = (
    "You are an experienced university lecturer who writes fair, clear quiz questions. "
    "You only ask about information that is explicitly stated in the lecture text you are given."
)

PROMPT = """TASK: GENERATE_QUIZ
Write {n} quiz questions about the lecture below.
Mix question types: "mcq" (exactly 4 options, one correct) and "short_answer".
Vary difficulty across "easy", "medium" and "hard".
For EVERY question, copy into "source_quote" the exact sentence from the lecture that supports the answer.
Reply with JSON only, no other text, in this format:
{{"questions": [{{"type": "mcq", "question": "...", "options": ["...", "...", "...", "..."],
"answer": "...", "source_quote": "...", "difficulty": "easy"}}]}}

LECTURE:
{lecture}"""


@dataclass
class Question:
    question: str
    answer: str
    type: str = "short_answer"
    options: list[str] = field(default_factory=list)
    source_quote: str = ""
    difficulty: str = "medium"
    grounded: bool = False
    grounding_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def extract_json(text: str) -> dict:
    """Parse the first JSON object in a model reply (tolerates code fences and chatter)."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model reply")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("Unbalanced JSON in model reply")


def generate_quiz(llm: LLM, lecture: str, n: int = 5, threshold: float = 0.8,
                  keep_ungrounded: bool = False) -> list[Question]:
    """Generate up to ``n`` questions and attach a grounding check to each."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": PROMPT.format(n=n, lecture=truncate(lecture))}]
    data = extract_json(llm.chat(messages, max_tokens=1500))
    questions: list[Question] = []
    for raw in data.get("questions", []):
        if not raw.get("question") or not raw.get("answer"):
            continue
        q = Question(question=str(raw["question"]).strip(), answer=str(raw["answer"]).strip(),
                     type=raw.get("type", "short_answer"),
                     options=[str(o) for o in raw.get("options", []) or []],
                     source_quote=str(raw.get("source_quote", "")).strip(),
                     difficulty=raw.get("difficulty", "medium"))
        if q.type == "mcq" and (len(q.options) < 2 or q.answer not in q.options):
            q.type = "short_answer"  # malformed MCQ: fall back rather than show a broken question
            q.options = []
        g = check_quote(q.source_quote, lecture, threshold)
        q.grounded, q.grounding_score = g.grounded, g.score
        if q.grounded or keep_ungrounded:
            questions.append(q)
    return questions[:n]
