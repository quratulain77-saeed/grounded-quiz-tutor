"""Conversational assessment: grade an answer, explain, and ask a follow-up."""

from __future__ import annotations

from dataclasses import dataclass

from .generator import Question, extract_json
from .llm import LLM

SYSTEM = (
    "You are a supportive tutor assessing a university student. Grade fairly against the expected answer "
    "and the lecture evidence. Accept answers that are correct in meaning even if worded differently. "
    "Keep feedback to two or three sentences, and base it only on the evidence given."
)

PROMPT = """TASK: GRADE
QUESTION: {question}
EXPECTED: {answer}
EVIDENCE FROM LECTURE: {evidence}
STUDENT ANSWER: {student}

Reply with JSON only:
{{"verdict": "correct" | "partially_correct" | "incorrect", "score": 0.0-1.0,
"feedback": "short explanation for the student",
"follow_up": "one follow-up question that checks deeper understanding of the same idea"}}"""


@dataclass
class Assessment:
    verdict: str
    score: float
    feedback: str
    follow_up: str


def assess(llm: LLM, q: Question, student_answer: str) -> Assessment:
    if not student_answer.strip():
        return Assessment("incorrect", 0.0, "No answer was given.", "")
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": PROMPT.format(question=q.question, answer=q.answer,
                                                          evidence=q.source_quote, student=student_answer)}]
    try:
        d = extract_json(llm.chat(messages, max_tokens=400))
    except ValueError:
        return Assessment("ungraded", 0.0, "The model reply could not be parsed; please try again.", "")
    verdict = str(d.get("verdict", "incorrect"))
    try:
        score = max(0.0, min(1.0, float(d.get("score", 0.0))))
    except (TypeError, ValueError):
        score = 0.0
    return Assessment(verdict, score, str(d.get("feedback", "")), str(d.get("follow_up", "")))
