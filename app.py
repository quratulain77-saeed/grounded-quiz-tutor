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
    state = {"questions": [q.to_dict() for q in kept], "i": 0, "mode": "main", "scores": [], "results": [],
             "backend": backend, "model": model}
    radio, box = _widgets(state)
    msg = "Quiz ready: open the **2 · Take the quiz** tab." if kept else ""
    return (_quiz_table(kept, len(all_qs) - len(kept)) + ("\n\n" + msg if msg else ""), state,
            _render_question(state), "", gr.Button(visible=bool(kept)), radio, box)


def _render_question(state) -> str:
    qs = state.get("questions", [])
    if not qs:
        return "Generate a quiz first (tab 1)."
    if state["i"] >= len(qs):
        total = sum(state["scores"])
        rows = ["| # | Question | Result |", "|---|---|---|"]
        for j, (q, r) in enumerate(zip(qs, state["results"]), 1):
            rows.append(f"| {j} | {q['question']} | {r} |")
        return f"### Quiz finished\nScore: **{total:.1f} / {len(qs)}**\n\n" + "\n".join(rows)
    q = qs[state["i"]]
    head = f"### Question {state['i'] + 1} of {len(qs)}  ·  {q['difficulty']}\n\n{q['question']}"
    if state["mode"] == "follow_up":
        head += f"\n\n**Follow-up:** {state['follow_up']}\n\n*Type your answer below.*"
    elif q["type"] == "mcq" and q["options"]:
        head += "\n\n*Choose one option, then press **Submit answer**.*"
    else:
        head += "\n\n*Type your answer, then press **Submit answer**.*"
    return head


def _widgets(state):
    """Show clickable options for MCQs, a text box otherwise."""
    qs = state.get("questions", [])
    finished = not qs or state["i"] >= len(qs)
    if finished:
        return gr.Radio(choices=[], value=None, visible=False), gr.Textbox(value="", visible=False)
    q = qs[state["i"]]
    is_mcq = q["type"] == "mcq" and q["options"] and state["mode"] == "main"
    return (gr.Radio(choices=q["options"] if is_mcq else [], value=None, visible=bool(is_mcq), label="Options"),
            gr.Textbox(value="", visible=not is_mcq, label="Your answer"))


def _view(state, feedback=""):
    radio, box = _widgets(state)
    return feedback, state, _render_question(state), radio, box


def on_answer(choice, text, state):
    if not state or not state.get("questions") or state["i"] >= len(state["questions"]):
        return _view(state or {"questions": []}, "Generate a quiz first, or start a new one.")
    if state["mode"] == "done":
        return _view(state, "Press **Next question** to continue.")
    q = Question(**state["questions"][state["i"]])
    answer = (choice or "") if (q.type == "mcq" and q.options and state["mode"] == "main") else (text or "")
    if not answer.strip():
        return _view(state, "⚠️ Please choose or type an answer first.")
    try:
        llm = _llm(state["backend"], state["model"])
        if state["mode"] == "main":
            a = assess(llm, q, answer)
            if q.type == "mcq" and q.options:  # exact, reliable scoring for multiple choice
                ok = answer.strip() == q.answer.strip()
                a.verdict, a.score = ("correct", 1.0) if ok else ("incorrect", 0.0)
                if not ok:
                    a.feedback = f"The correct answer is: **{q.answer}**. " + a.feedback
            state["scores"].append(a.score)
            state["results"].append(f"{a.verdict.replace('_', ' ')} ({a.score:.1f})")
            fb = (f"## {a.verdict.replace('_', ' ').title()}  (score {a.score:.1f})\n\n{a.feedback}\n\n"
                  f"> **Evidence from the lecture:** *{q.source_quote}*")
            if a.follow_up:
                state["mode"], state["follow_up"] = "follow_up", a.follow_up
                fb += "\n\nAnswer the **follow-up question** above, or press **Next question**."
            else:
                state["mode"] = "done"
        else:
            fq = Question(question=state["follow_up"], answer="(open question: judge using the evidence)",
                          source_quote=q.source_quote)
            a = assess(llm, fq, answer)
            fb = f"## Follow-up: {a.verdict.replace('_', ' ')}\n\n{a.feedback}\n\nPress **Next question** to continue."
            state["mode"] = "done"
    except Exception as e:  # show problems on the page instead of failing silently
        return _view(state, f"❌ Grading failed: `{type(e).__name__}: {e}`. Check the Colab output for details.")
    return _view(state, fb)


def on_next(state):
    note = ""
    if state and state.get("questions") and state["i"] < len(state["questions"]):
        if state["mode"] == "main":
            state["scores"].append(0.0)
            state["results"].append("skipped")
            note = "Previous question skipped (0 points)."
        state["i"] += 1
        state["mode"] = "main"
    return _view(state or {"questions": []}, note)


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
        q_md = gr.Markdown("Generate a quiz first (tab 1).")
        choice = gr.Radio(choices=[], label="Options", visible=False)
        ans = gr.Textbox(label="Your answer", lines=2)
        with gr.Row():
            submit = gr.Button("Submit answer", variant="primary")
            nxt = gr.Button("Next question")
        fb_md = gr.Markdown()

    gen.click(on_generate, [lecture_file, lecture_text, backend, model, n, keep],
              [table, state, q_md, fb_md, export_btn, choice, ans])
    submit.click(lambda: "⏳ Grading your answer…", None, fb_md, queue=False).then(
        on_answer, [choice, ans, state], [fb_md, state, q_md, choice, ans])
    nxt.click(on_next, [state], [fb_md, state, q_md, choice, ans])
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
