"""Gradio web app for Grounded Quiz Tutor.

Run:  python app.py      (then open http://127.0.0.1:7860)
"""

from __future__ import annotations

import json
import os
import tempfile

import gradio as gr

from quiztutor.generator import Question, generate_quiz
from quiztutor.llm import make_llm
from quiztutor.text import load_text
from quiztutor.tutor import assess

BACKENDS = {"Open model on this machine / Colab GPU": "local", "Hugging Face Inference API": "hf",
            "Local Ollama / OpenAI-compatible": "ollama", "Offline demo (no model)": "mock"}
DEFAULT_BACKEND = os.getenv("QT_DEFAULT_BACKEND", "Offline demo (no model)")
EXAMPLE = os.path.join(os.path.dirname(__file__), "examples", "sample_lecture.txt")


def _llm(backend_label: str, model: str):
    kwargs = {"model": model.strip()} if model.strip() and BACKENDS[backend_label] != "mock" else {}
    return make_llm(BACKENDS[backend_label], **kwargs)


def _quiz_table(qs: list[Question], rejected: int) -> str:
    if not qs:
        return "No grounded questions were produced. Try a longer lecture text or another model."
    rows = ["| # | Question | Type | Difficulty | Evidence check |", "|---|---|---|---|---|"]
    for i, q in enumerate(qs, 1):
        mark = f"✅ grounded ({q.grounding_score:.2f})" if q.grounded else f"⚠️ not found ({q.grounding_score:.2f})"
        rows.append(f"| {i} | {q.question} | {q.type} | {q.difficulty} | {mark} |")
    note = f"\n\n**{rejected} question(s) removed** because their quoted evidence was not found in the lecture." \
        if rejected else ""
    return "\n".join(rows) + note


def on_generate(file, text, backend, model, n, keep):
    lecture = load_text(file) if file else (text or "")
    if len(lecture.split()) < 40:
        raise gr.Error("Please provide at least a paragraph of lecture text (40+ words).")
    llm = _llm(backend, model)
    all_qs = generate_quiz(llm, lecture, n=int(n), keep_ungrounded=True)
    kept = all_qs if keep else [q for q in all_qs if q.grounded]
    state = {"questions": [q.to_dict() for q in kept], "i": 0, "mode": "main", "scores": [],
             "backend": backend, "model": model}
    first = _render_question(state)
    return _quiz_table(kept, len(all_qs) - len(kept)), state, first, "", gr.update(visible=bool(kept))


def _render_question(state) -> str:
    qs = state["questions"]
    if not qs:
        return ""
    if state["i"] >= len(qs):
        total = sum(state["scores"])
        return f"### Quiz finished\nScore: **{total:.1f} / {len(qs)}**"
    q = qs[state["i"]]
    head = f"### Question {state['i'] + 1} of {len(qs)}  ·  {q['difficulty']}\n\n{q['question']}"
    if q["type"] == "mcq" and q["options"]:
        head += "\n\n" + "\n".join(f"- {o}" for o in q["options"])
    if state["mode"] == "follow_up":
        head += f"\n\n**Follow-up:** {state['follow_up']}"
    return head


def on_answer(answer, state):
    if not state or not state.get("questions") or state["i"] >= len(state["questions"]):
        return "", state, _render_question(state or {"questions": []}), ""
    if state["mode"] == "done":
        return "Press **Next question** to continue.", state, _render_question(state), answer
    q = Question(**state["questions"][state["i"]])
    llm = _llm(state["backend"], state["model"])
    if state["mode"] == "main":
        a = assess(llm, q, answer)
        state["scores"].append(a.score)
        fb = f"**{a.verdict.replace('_', ' ').title()}** (score {a.score:.1f})\n\n{a.feedback}\n\n" \
             f"> Evidence from the lecture: *{q.source_quote}*"
        if a.follow_up:
            state["mode"], state["follow_up"] = "follow_up", a.follow_up
            fb += "\n\nAnswer the follow-up below, or press **Next question**."
    else:
        fq = Question(question=state["follow_up"], answer="(open question: judge using the evidence)",
                      source_quote=q.source_quote)
        a = assess(llm, fq, answer)
        fb = f"**Follow-up: {a.verdict.replace('_', ' ')}**\n\n{a.feedback}"
        state["mode"] = "done"
    return fb, state, _render_question(state), ""


def on_next(state):
    if state and state.get("questions"):
        if state["mode"] == "main" and state["i"] < len(state["questions"]):
            state["scores"].append(0.0)  # skipped
        state["i"] += 1
        state["mode"] = "main"
    return state, _render_question(state or {"questions": []}), "", ""


def on_export(state):
    if not state or not state.get("questions"):
        raise gr.Error("Generate a quiz first.")
    path = os.path.join(tempfile.mkdtemp(), "quiz.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"questions": state["questions"]}, f, ensure_ascii=False, indent=2)
    return path


with gr.Blocks(title="Grounded Quiz Tutor") as demo:
    gr.Markdown("# Grounded Quiz Tutor\nTurn lecture notes into quiz questions with an open LLM, check that every "
                "question is supported by the lecture, then practise with an AI tutor that grades your answers "
                "and asks follow-up questions.")
    state = gr.State({})
    with gr.Tab("1 · Lecture"):
        with gr.Row():
            backend = gr.Dropdown(list(BACKENDS), value=DEFAULT_BACKEND, label="Model back-end")
            model = gr.Textbox(label="Model name (optional)", placeholder="e.g. Qwen/Qwen2.5-7B-Instruct")
            n = gr.Slider(3, 10, value=5, step=1, label="Number of questions")
        lecture_file = gr.File(label="Upload lecture (.txt, .md or .pdf)", file_types=[".txt", ".md", ".pdf"],
                               type="filepath")
        lecture_text = gr.Textbox(label="…or paste lecture text", lines=10,
                                  value=open(EXAMPLE, encoding="utf-8").read() if os.path.exists(EXAMPLE) else "")
        keep = gr.Checkbox(label="Keep questions whose evidence could not be verified (not recommended)")
        gen = gr.Button("Generate quiz", variant="primary")
        table = gr.Markdown()
        export_btn = gr.Button("Export quiz as JSON", visible=False)
        export_file = gr.File(label="Download")
    with gr.Tab("2 · Take the quiz"):
        q_md = gr.Markdown("Generate a quiz first.")
        ans = gr.Textbox(label="Your answer", lines=2)
        with gr.Row():
            submit = gr.Button("Submit answer", variant="primary")
            nxt = gr.Button("Next question")
        fb_md = gr.Markdown()

    gen.click(on_generate, [lecture_file, lecture_text, backend, model, n, keep],
              [table, state, q_md, fb_md, export_btn])
    submit.click(on_answer, [ans, state], [fb_md, state, q_md, ans])
    nxt.click(on_next, [state], [state, q_md, fb_md, ans])
    export_btn.click(on_export, [state], [export_file])

def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    # In Colab a public link is needed to open the app; elsewhere it runs on http://127.0.0.1:7860
    share = _in_colab() or os.getenv("QT_SHARE") == "1"
    demo.queue().launch(share=share, debug=_in_colab())
