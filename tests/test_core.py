import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quiztutor.generator import Question, extract_json, generate_quiz  # noqa: E402
from quiztutor.grounding import check_quote  # noqa: E402
from quiztutor.llm import MockLLM  # noqa: E402
from quiztutor.text import split_sentences  # noqa: E402
from quiztutor.tutor import assess  # noqa: E402

LECTURE = (Path(__file__).resolve().parents[1] / "examples" / "sample_lecture.txt").read_text()


def test_split_sentences():
    sents = split_sentences("First sentence here. Second one! Third? Yes.")
    assert sents == ["First sentence here.", "Second one!", "Third?", "Yes."]


def test_grounded_quote_is_found():
    r = check_quote("Roman Urdu has no standard spelling", LECTURE)
    assert r.grounded and r.score == 1.0


def test_paraphrased_quote_with_small_changes_is_found():
    r = check_quote("Tokenization is the process of splitting text into small units called tokens.", LECTURE)
    assert r.grounded and r.score >= 0.8


def test_hallucinated_quote_is_flagged():
    r = check_quote("Transformers were invented in 1950 by Alan Turing.", LECTURE)
    assert not r.grounded


def test_extract_json_handles_code_fences_and_chatter():
    reply = 'Sure! Here you go:\n```json\n{"questions": [{"question": "Q?", "answer": "A"}]}\n```'
    assert extract_json(reply)["questions"][0]["answer"] == "A"


def test_generate_quiz_filters_ungrounded_questions():
    fake = {"questions": [
        {"type": "short_answer", "question": "What does BPE merge?", "answer": "the most frequent pair",
         "source_quote": "Byte Pair Encoding (BPE) starts from individual characters and repeatedly merges the "
                         "most frequent pair of symbols into a new symbol.", "difficulty": "easy"},
        {"type": "short_answer", "question": "Who invented BPE?", "answer": "Shakespeare",
         "source_quote": "BPE was invented by Shakespeare in 1600.", "difficulty": "hard"},
        {"type": "mcq", "question": "Which model uses WordPiece?", "options": ["GPT", "BERT", "ELMo", "T5"],
         "answer": "BERT", "source_quote": "WordPiece, used by BERT, follows a similar idea",
         "difficulty": "medium"},
    ]}
    llm = MockLLM(lambda m: json.dumps(fake))
    qs = generate_quiz(llm, LECTURE, n=5)
    assert [q.question for q in qs] == ["What does BPE merge?", "Which model uses WordPiece?"]
    assert all(q.grounded for q in qs)
    kept_all = generate_quiz(llm, LECTURE, n=5, keep_ungrounded=True)
    assert len(kept_all) == 3 and not kept_all[1].grounded


def test_malformed_mcq_falls_back_to_short_answer():
    fake = {"questions": [{"type": "mcq", "question": "Q?", "options": ["x", "y"], "answer": "z",
                           "source_quote": "Roman Urdu has no standard spelling"}]}
    qs = generate_quiz(MockLLM(lambda m: json.dumps(fake)), LECTURE)
    assert qs[0].type == "short_answer" and qs[0].options == []


def test_default_mock_runs_end_to_end():
    llm = MockLLM()
    qs = generate_quiz(llm, LECTURE, n=3)
    assert len(qs) == 3 and all(q.grounded for q in qs)
    a = assess(llm, qs[0], qs[0].answer)
    assert a.verdict == "correct" and a.follow_up


def test_assess_handles_unparseable_reply():
    q = Question(question="Q?", answer="A", source_quote="x")
    a = assess(MockLLM(lambda m: "I am not JSON"), q, "something")
    assert a.verdict == "ungraded"


def test_assess_clamps_score():
    q = Question(question="Q?", answer="A", source_quote="x")
    a = assess(MockLLM(lambda m: '{"verdict": "correct", "score": 7, "feedback": "f", "follow_up": ""}'), q, "A")
    assert a.score == 1.0
