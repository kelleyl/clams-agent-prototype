"""
QA Review & ELO Evaluation App

Gradio web interface for:
  1. HITL review of generated QA pairs (approve/reject/edit)
  2. ELO-style pairwise comparison of model answers

Usage:
    python review_app.py                              # Launch review UI
    python review_app.py --qa-file raw/index_qa.jsonl # Specific QA file
    python review_app.py --port 7861                  # Custom port

Data is persisted to qa-data/reviews/ as JSONL files.
"""

import json
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr

# ── Data persistence ─────────────────────────────────────────────

REVIEWS_DIR = Path(__file__).parent / "reviews"
REVIEWS_DIR.mkdir(exist_ok=True)

REVIEW_FILE = REVIEWS_DIR / "review_decisions.jsonl"
ELO_FILE = REVIEWS_DIR / "elo_comparisons.jsonl"
ELO_RATINGS_FILE = REVIEWS_DIR / "elo_ratings.json"


def load_qa_entries(path: Path) -> list[dict]:
    """Load QA entries from JSONL file."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def save_review(entry_id: str, decision: str, edited_question: str = "",
                edited_answer: str = "", notes: str = ""):
    """Append a review decision to the reviews file."""
    record = {
        "entry_id": entry_id,
        "decision": decision,
        "edited_question": edited_question if edited_question else None,
        "edited_answer": edited_answer if edited_answer else None,
        "notes": notes if notes else None,
        "timestamp": datetime.now().isoformat(),
    }
    with open(REVIEW_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_reviews() -> dict[str, dict]:
    """Load all review decisions, keyed by entry_id (latest wins)."""
    reviews = {}
    if REVIEW_FILE.exists():
        with open(REVIEW_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    reviews[record["entry_id"]] = record
    return reviews


def save_elo_comparison(entry_id: str, model_a: str, model_b: str,
                        answer_a: str, answer_b: str, winner: str):
    """Save an ELO pairwise comparison."""
    record = {
        "entry_id": entry_id,
        "model_a": model_a,
        "model_b": model_b,
        "answer_a": answer_a,
        "answer_b": answer_b,
        "winner": winner,  # "A", "B", or "tie"
        "timestamp": datetime.now().isoformat(),
    }
    with open(ELO_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_elo_comparisons() -> list[dict]:
    """Load all ELO comparisons."""
    comparisons = []
    if ELO_FILE.exists():
        with open(ELO_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    comparisons.append(json.loads(line))
    return comparisons


# ── ELO calculation ──────────────────────────────────────────────

DEFAULT_ELO = 1500
K_FACTOR = 32


def compute_elo_ratings(comparisons: list[dict]) -> dict[str, float]:
    """Compute ELO ratings from pairwise comparisons."""
    ratings: dict[str, float] = {}

    for comp in comparisons:
        model_a = comp["model_a"]
        model_b = comp["model_b"]
        winner = comp["winner"]

        if model_a not in ratings:
            ratings[model_a] = DEFAULT_ELO
        if model_b not in ratings:
            ratings[model_b] = DEFAULT_ELO

        ra, rb = ratings[model_a], ratings[model_b]
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        eb = 1 / (1 + 10 ** ((ra - rb) / 400))

        if winner == "A":
            sa, sb = 1.0, 0.0
        elif winner == "B":
            sa, sb = 0.0, 1.0
        else:  # tie
            sa, sb = 0.5, 0.5

        ratings[model_a] = ra + K_FACTOR * (sa - ea)
        ratings[model_b] = rb + K_FACTOR * (sb - eb)

    return ratings


def save_elo_ratings(ratings: dict[str, float]):
    """Save computed ELO ratings."""
    with open(ELO_RATINGS_FILE, "w") as f:
        json.dump(ratings, f, indent=2)


# ── Format helpers ───────────────────────────────────────────────

def format_answer(answer: dict) -> str:
    """Format an answer dict for display."""
    parts = []

    if "mc" in answer:
        mc = answer["mc"]
        for letter, choice in mc["choices"].items():
            marker = " <<< CORRECT" if letter == mc["correct"] else ""
            parts.append(f"  {letter}) {choice}{marker}")
        return "Multiple Choice:\n" + "\n".join(parts)

    if "scene_label" in answer:
        parts.append(f"Scene: {answer['scene_label']}")
    if "exists" in answer:
        parts.append(f"Exists: {answer['exists']}")
    if "count" in answer:
        parts.append(f"Count: {answer['count']}")
    if "text" in answer:
        text = answer["text"]
        if len(text) > 300:
            text = text[:300] + "..."
        parts.append(f"Text: {text}")
    if "value" in answer:
        parts.append(f"Value: {answer['value']}")
    if "explanation" in answer:
        parts.append(f"Explanation: {answer['explanation']}")
    if "people" in answer:
        parts.append(f"People: {', '.join(answer['people'][:10])}")
    if "organizations" in answer:
        parts.append(f"Organizations: {', '.join(answer['organizations'][:10])}")
    if "locations" in answer:
        parts.append(f"Locations: {', '.join(answer['locations'][:10])}")
    if "entities" in answer and not parts:
        ents = [e["text"] for e in answer["entities"][:10]]
        parts.append(f"Entities: {', '.join(ents)}")
    if "entity" in answer and isinstance(answer["entity"], str):
        parts.append(f"Entity: {answer['entity']}")

    return "\n".join(parts) if parts else json.dumps(answer, indent=2)[:500]


def format_context(ctx: dict | None) -> str:
    """Format context dict for display."""
    if not ctx:
        return "No context"
    parts = []
    if "video_guid" in ctx:
        parts.append(f"Video: {ctx['video_guid']}")
    if "timestamp" in ctx:
        parts.append(f"Timestamp: {ctx['timestamp']}")
    if "time_range" in ctx:
        tr = ctx["time_range"]
        parts.append(f"Range: {tr.get('start', '?')} - {tr.get('end', '?')}")
    if "transcript_snippet" in ctx:
        parts.append(f"Transcript: {ctx['transcript_snippet'][:150]}...")
    return "\n".join(parts) if parts else json.dumps(ctx, indent=2)[:300]


# ── Review Tab ───────────────────────────────────────────────────

class ReviewState:
    """Manages state for the review interface."""

    def __init__(self, entries: list[dict]):
        self.entries = entries
        self.reviews = load_reviews()
        self.current_idx = 0
        self._find_next_unreviewed()

    def _find_next_unreviewed(self):
        """Advance to next unreviewed entry."""
        start = self.current_idx
        for i in range(len(self.entries)):
            idx = (start + i) % len(self.entries)
            if self.entries[idx]["id"] not in self.reviews:
                self.current_idx = idx
                return
        # All reviewed — stay at current

    @property
    def current(self) -> dict:
        return self.entries[self.current_idx]

    @property
    def progress(self) -> str:
        reviewed = len(self.reviews)
        total = len(self.entries)
        return f"{reviewed}/{total} reviewed ({reviewed/total*100:.0f}%)"

    def submit_review(self, decision: str, edited_q: str = "", edited_a: str = "",
                      notes: str = "") -> dict:
        """Submit review and advance."""
        entry = self.current
        save_review(entry["id"], decision, edited_q, edited_a, notes)
        self.reviews[entry["id"]] = {"decision": decision}
        self.current_idx = (self.current_idx + 1) % len(self.entries)
        self._find_next_unreviewed()
        return self.current

    def go_to(self, idx: int) -> dict:
        """Jump to a specific entry."""
        self.current_idx = max(0, min(idx, len(self.entries) - 1))
        return self.current

    def stats(self) -> dict[str, int]:
        """Get review stats by decision type."""
        counts: dict[str, int] = {"approved": 0, "rejected": 0, "edited": 0, "skipped": 0}
        for r in self.reviews.values():
            d = r.get("decision", "")
            if d in counts:
                counts[d] += 1
        return counts


# ── Build Gradio UI ──────────────────────────────────────────────

def build_app(qa_file: Path) -> gr.Blocks:
    entries = load_qa_entries(qa_file)
    review_state = ReviewState(entries)

    def get_display(entry: dict) -> tuple:
        """Return display values for an entry."""
        review_status = review_state.reviews.get(entry["id"], {}).get("decision", "unreviewed")
        status_icon = {"approved": "Approved", "rejected": "Rejected",
                       "edited": "Edited", "skipped": "Skipped"}.get(review_status, "Unreviewed")
        return (
            f"**{entry['id']}** | {entry['category']} | {entry.get('difficulty', '?')} | Status: {status_icon}",
            entry["question"],
            format_answer(entry["answer"]),
            format_context(entry.get("context")),
            ", ".join(entry.get("tags", [])),
            review_state.progress,
        )

    with gr.Blocks(title="CLAMS QA Review") as app:
        gr.Markdown("# CLAMS QA Review & ELO Evaluation")

        with gr.Tabs():
            # ── Tab 1: Review ──
            with gr.Tab("Review QA Pairs"):
                with gr.Row():
                    progress_display = gr.Textbox(label="Progress", interactive=False)
                    entry_slider = gr.Slider(
                        minimum=0, maximum=len(entries) - 1, step=1, value=0,
                        label="Navigate to entry #"
                    )

                header_display = gr.Markdown()

                with gr.Row():
                    with gr.Column(scale=2):
                        question_display = gr.Textbox(label="Question", lines=3, interactive=False)
                        answer_display = gr.Textbox(label="Gold Answer", lines=8, interactive=False)
                    with gr.Column(scale=1):
                        context_display = gr.Textbox(label="Context", lines=5, interactive=False)
                        tags_display = gr.Textbox(label="Tags", interactive=False)

                gr.Markdown("### Review Decision")
                with gr.Row():
                    edited_question = gr.Textbox(label="Edited Question (optional)", lines=2)
                    edited_answer = gr.Textbox(label="Edited Answer (optional)", lines=2)
                notes_input = gr.Textbox(label="Notes (optional)", lines=1)

                with gr.Row():
                    approve_btn = gr.Button("Approve", variant="primary")
                    reject_btn = gr.Button("Reject", variant="stop")
                    edit_btn = gr.Button("Save Edit", variant="secondary")
                    skip_btn = gr.Button("Skip")

                # Stats
                with gr.Accordion("Review Statistics", open=False):
                    stats_display = gr.JSON(label="Stats")
                    refresh_stats_btn = gr.Button("Refresh Stats")

                # Wire up review actions
                all_outputs = [header_display, question_display, answer_display,
                               context_display, tags_display, progress_display]

                def do_approve(eq, ea, notes):
                    review_state.submit_review("approved", eq, ea, notes)
                    return get_display(review_state.current) + ("", "", "")

                def do_reject(eq, ea, notes):
                    review_state.submit_review("rejected", eq, ea, notes)
                    return get_display(review_state.current) + ("", "", "")

                def do_edit(eq, ea, notes):
                    review_state.submit_review("edited", eq, ea, notes)
                    return get_display(review_state.current) + ("", "", "")

                def do_skip(eq, ea, notes):
                    review_state.submit_review("skipped", eq, ea, notes)
                    return get_display(review_state.current) + ("", "", "")

                def do_navigate(idx):
                    review_state.go_to(int(idx))
                    return get_display(review_state.current)

                clear_outputs = all_outputs + [edited_question, edited_answer, notes_input]

                approve_btn.click(do_approve, [edited_question, edited_answer, notes_input], clear_outputs)
                reject_btn.click(do_reject, [edited_question, edited_answer, notes_input], clear_outputs)
                edit_btn.click(do_edit, [edited_question, edited_answer, notes_input], clear_outputs)
                skip_btn.click(do_skip, [edited_question, edited_answer, notes_input], clear_outputs)
                entry_slider.change(do_navigate, [entry_slider], all_outputs)

                def get_stats():
                    stats = review_state.stats()
                    stats["total"] = len(entries)
                    stats["remaining"] = stats["total"] - sum(v for k, v in stats.items() if k != "total")
                    return stats

                refresh_stats_btn.click(get_stats, outputs=[stats_display])

            # ── Tab 2: ELO Comparison ──
            with gr.Tab("ELO Model Comparison"):
                gr.Markdown("""
                ### Pairwise Model Answer Comparison

                Compare answers from two models on the same question.
                Your judgments build ELO ratings for each model.

                **Workflow:**
                1. Load model answer files (JSONL with `id`, `model`, `answer` fields)
                2. See a random question with both model answers side-by-side
                3. Pick the better answer, or declare a tie
                """)

                with gr.Row():
                    model_a_file = gr.File(label="Model A Answers (JSONL)", file_types=[".jsonl"])
                    model_b_file = gr.File(label="Model B Answers (JSONL)", file_types=[".jsonl"])
                load_models_btn = gr.Button("Load Model Answers")

                elo_question = gr.Textbox(label="Question", lines=3, interactive=False)
                elo_gold = gr.Textbox(label="Gold Answer", lines=3, interactive=False)

                with gr.Row():
                    with gr.Column():
                        elo_label_a = gr.Markdown("### Model A")
                        elo_answer_a = gr.Textbox(label="Answer A", lines=6, interactive=False)
                    with gr.Column():
                        elo_label_b = gr.Markdown("### Model B")
                        elo_answer_b = gr.Textbox(label="Answer B", lines=6, interactive=False)

                with gr.Row():
                    pick_a_btn = gr.Button("A is Better", variant="primary")
                    pick_tie_btn = gr.Button("Tie")
                    pick_b_btn = gr.Button("B is Better", variant="primary")

                elo_status = gr.Textbox(label="Status", interactive=False)

                with gr.Accordion("ELO Leaderboard", open=True):
                    elo_leaderboard = gr.JSON(label="Ratings")
                    refresh_elo_btn = gr.Button("Refresh Leaderboard")

                # ELO state
                elo_state = gr.State({
                    "model_a_answers": {},
                    "model_b_answers": {},
                    "model_a_name": "",
                    "model_b_name": "",
                    "current_entry_id": "",
                    "pairs": [],
                })

                def load_model_answers(file_a, file_b, state):
                    if not file_a or not file_b:
                        return state, "Please upload both model answer files.", "", "", "", "", "", ""

                    answers_a = {}
                    answers_b = {}
                    name_a = "Model A"
                    name_b = "Model B"

                    with open(file_a.name) as f:
                        for line in f:
                            r = json.loads(line.strip())
                            answers_a[r["id"]] = r.get("answer", "")
                            if "model" in r:
                                name_a = r["model"]

                    with open(file_b.name) as f:
                        for line in f:
                            r = json.loads(line.strip())
                            answers_b[r["id"]] = r.get("answer", "")
                            if "model" in r:
                                name_b = r["model"]

                    # Find common entries
                    common_ids = list(set(answers_a.keys()) & set(answers_b.keys()))
                    if not common_ids:
                        return state, "No overlapping entry IDs found.", "", "", "", "", f"### {name_a}", f"### {name_b}"

                    random.shuffle(common_ids)
                    state = {
                        "model_a_answers": answers_a,
                        "model_b_answers": answers_b,
                        "model_a_name": name_a,
                        "model_b_name": name_b,
                        "current_entry_id": "",
                        "pairs": common_ids,
                    }

                    # Show first pair
                    return show_next_pair(state)

                def show_next_pair(state):
                    pairs = state["pairs"]
                    if not pairs:
                        return state, "All pairs compared!", "", "", "", "", f"### {state['model_a_name']}", f"### {state['model_b_name']}"

                    entry_id = pairs.pop(0)
                    state["current_entry_id"] = entry_id

                    # Find the question from loaded entries
                    question = ""
                    gold = ""
                    for e in entries:
                        if e["id"] == entry_id:
                            question = e["question"]
                            gold = format_answer(e["answer"])
                            break

                    ans_a = state["model_a_answers"].get(entry_id, "(no answer)")
                    ans_b = state["model_b_answers"].get(entry_id, "(no answer)")

                    if isinstance(ans_a, dict):
                        ans_a = json.dumps(ans_a, indent=2)
                    if isinstance(ans_b, dict):
                        ans_b = json.dumps(ans_b, indent=2)

                    remaining = len(pairs)
                    status = f"Comparing: {entry_id} | {remaining} remaining"

                    return (state, status, question, gold, ans_a, ans_b,
                            f"### {state['model_a_name']}", f"### {state['model_b_name']}")

                def pick_winner(winner: str, state):
                    entry_id = state.get("current_entry_id", "")
                    if not entry_id:
                        return state, "No comparison loaded.", "", "", "", "", "", ""

                    save_elo_comparison(
                        entry_id,
                        state["model_a_name"],
                        state["model_b_name"],
                        str(state["model_a_answers"].get(entry_id, "")),
                        str(state["model_b_answers"].get(entry_id, "")),
                        winner,
                    )
                    return show_next_pair(state)

                elo_model_outputs = [elo_state, elo_status, elo_question, elo_gold,
                                     elo_answer_a, elo_answer_b, elo_label_a, elo_label_b]

                load_models_btn.click(load_model_answers,
                                      [model_a_file, model_b_file, elo_state],
                                      elo_model_outputs)
                pick_a_btn.click(lambda s: pick_winner("A", s), [elo_state], elo_model_outputs)
                pick_b_btn.click(lambda s: pick_winner("B", s), [elo_state], elo_model_outputs)
                pick_tie_btn.click(lambda s: pick_winner("tie", s), [elo_state], elo_model_outputs)

                def refresh_leaderboard():
                    comparisons = load_elo_comparisons()
                    if not comparisons:
                        return {"message": "No comparisons yet"}
                    ratings = compute_elo_ratings(comparisons)
                    save_elo_ratings(ratings)
                    # Sort by rating descending
                    sorted_ratings = dict(sorted(ratings.items(), key=lambda x: -x[1]))
                    result = {
                        "ratings": {k: round(v, 1) for k, v in sorted_ratings.items()},
                        "total_comparisons": len(comparisons),
                    }
                    return result

                refresh_elo_btn.click(refresh_leaderboard, outputs=[elo_leaderboard])

            # ── Tab 3: Dataset Stats ──
            with gr.Tab("Dataset Overview"):
                def compute_overview():
                    by_cat = {}
                    by_diff = {}
                    by_tag = {}
                    mc_count = 0
                    for e in entries:
                        cat = e.get("category", "?")
                        by_cat[cat] = by_cat.get(cat, 0) + 1
                        diff = e.get("difficulty", "?")
                        by_diff[diff] = by_diff.get(diff, 0) + 1
                        for tag in e.get("tags", []):
                            by_tag[tag] = by_tag.get(tag, 0) + 1
                        if "mc" in e.get("answer", {}):
                            mc_count += 1

                    reviews = load_reviews()
                    review_stats = {"approved": 0, "rejected": 0, "edited": 0, "skipped": 0}
                    for r in reviews.values():
                        d = r.get("decision", "")
                        if d in review_stats:
                            review_stats[d] += 1

                    return {
                        "total_entries": len(entries),
                        "multiple_choice": mc_count,
                        "free_form": len(entries) - mc_count,
                        "by_category": dict(sorted(by_cat.items())),
                        "by_difficulty": by_diff,
                        "top_tags": dict(sorted(by_tag.items(), key=lambda x: -x[1])[:15]),
                        "review_progress": review_stats,
                    }

                overview_json = gr.JSON(label="Dataset Overview")
                refresh_overview_btn = gr.Button("Refresh Overview")
                refresh_overview_btn.click(compute_overview, outputs=[overview_json])

        # Initialize display on load
        app.load(lambda: get_display(review_state.current), outputs=all_outputs)

    return app


def main():
    import argparse

    parser = argparse.ArgumentParser(description="QA Review & ELO App")
    parser.add_argument("--qa-file", default=None, help="Path to QA JSONL file")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    # Find QA file
    if args.qa_file:
        qa_path = Path(args.qa_file)
    else:
        # Try MC first, then regular
        qa_dir = Path(__file__).parent / "raw"
        mc_path = qa_dir / "index_qa_mc.jsonl"
        regular_path = qa_dir / "index_qa.jsonl"
        qa_path = mc_path if mc_path.exists() else regular_path

    if not qa_path.exists():
        print(f"QA file not found: {qa_path}")
        print("Run generate_index_qa.py first to create QA data.")
        return

    print(f"Loading QA data from: {qa_path}")
    app = build_app(qa_path)
    app.launch(server_port=args.port, share=args.share, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
