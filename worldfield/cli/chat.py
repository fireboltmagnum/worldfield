"""CLI output store and section rendering for the cognitive pipeline display."""
from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.text import Text
from prompt_toolkit.formatted_text import ANSI as PTK_ANSI

_rich_console = Console(color_system="truecolor", force_terminal=True, width=80)


def _render_ansi(renderable) -> str:
    with _rich_console.capture() as cap:
        _rich_console.print(renderable)
    return cap.get()


class Turn:
    """One user input + engine response, rendered into sections."""

    def __init__(self, user_text: str, engine_result: dict[str, Any]):
        self.user_text = user_text
        self.engine = engine_result
        self.sections: list[tuple[str, str]] = []  # (section_name, ansi_text)
        self._build()

    def _build(self):
        eng = self.engine
        timings = eng.get("timings", {})

        # -- INPUT --
        self.sections.append(("INPUT", _render_ansi(Text(eng.get("text", ""), style="white"))))

        # -- UNDERSTANDING --
        concepts = eng.get("concepts_extracted", [])
        rels = eng.get("extracted_relations_raw", [])
        if concepts:
            lines = [f"Concepts:  {', '.join(concepts)}"]
            for r in rels[:5]:
                src = r.get("source", "?")
                pred = r.get("predicate", "?")
                tgt = r.get("target", "?")
                lines.append(f"           {src} -[{pred}]? {tgt}")
            best_conf = max((r.get("confidence", 0) for r in rels), default=0.0)
            lines.append(f"Confidence: {best_conf:.2f}")
            t = timings.get("understanding", 0)
            header = f"UNDERSTANDING ({t:.0f}ms)" if t else "UNDERSTANDING"
            self.sections.append((header, _render_ansi(Text("\n".join(lines), style="cyan"))))

        # -- ACTIVATION --
        active = eng.get("activation_active", [])
        working_set = eng.get("activation_working_set", [])
        if active or working_set:
            rows = []
            bar_width = 16
            if active:
                rows.append("Active:")
                for name, level in active[:6]:
                    filled = int(min(level, 1.0) * bar_width)
                    bar = "█" * filled + "░" * (bar_width - filled)
                    rows.append(f"  {name:<16s} {bar}  {level:.3f}")
            if working_set:
                rows.append("Working Set:")
                for name, level in working_set[:4]:
                    filled = int(min(level, 1.0) * bar_width)
                    bar = "█" * filled + "░" * (bar_width - filled)
                    rows.append(f"  {name:<16s} {bar}  {level:.3f}")
            t = timings.get("activation", 0)
            header = f"ACTIVATION ({t:.0f}ms)" if t else "ACTIVATION"
            self.sections.append((header, _render_ansi(Text("\n".join(rows), style="green"))))

        # -- WORLD STATE --
        ws = eng.get("world_state", {})
        if ws and ws.get("entities"):
            lines_ws: list[str] = []
            entities = ws.get("entities", {})
            rels_ws = ws.get("relations", [])
            attrs_ws = ws.get("attributes", {})
            for name, conf in sorted(entities.items(), key=lambda x: -x[1])[:8]:
                attr_str = ""
                if name in attrs_ws:
                    attr_list = [f"{a}={v:.2f}" for a, v in attrs_ws[name].items()]
                    attr_str = f"  [{', '.join(attr_list[:3])}]"
                lines_ws.append(f"  {name:<14s} {conf:.2f}{attr_str}")
            if rels_ws:
                lines_ws.append("")
                for r in rels_ws[:6]:
                    src = r.get("source", "?")
                    pred = r.get("predicate", "?")
                    tgt = r.get("target", "?")
                    conf = r.get("confidence", 0.0)
                    neg = "  (¬)" if not r.get("polarity", True) else ""
                    lines_ws.append(f"  {src} -[{pred}]? {tgt}  {conf:.2f}{neg}")
            if ws.get("n_alternatives", 0) > 0:
                lines_ws.append("")
                lines_ws.append(f"  ⚠ {ws['n_alternatives']} competing interpretation(s)")
            t = timings.get("world_state", 0)
            header = f"WORLD STATE ({t:.0f}ms)" if t else "WORLD STATE"
            self.sections.append((header, _render_ansi(Text("\n".join(lines_ws), style="yellow"))))

        # -- WORLD MODEL UPDATE --
        pre_c, pre_r = eng.get("graph_pre_state", (0, 0))
        post_c = eng.get("total_concepts", 0)
        post_r = eng.get("total_relations", 0)
        new_c = post_c - pre_c
        new_r = post_r - pre_r
        if new_c > 0 or new_r > 0 or rels:
            lines2 = []
            for r in rels[:5]:
                src = r.get("source", "?")
                pred = r.get("predicate", "?")
                tgt = r.get("target", "?")
                lines2.append(f"  {src} -[{pred}]? {tgt}   (new)")
            if not lines2 and (new_c > 0 or new_r > 0):
                lines2.append(f"  +{new_c} concepts, +{new_r} relations stored")
            t = timings.get("world_update", 0)
            header = f"WORLD MODEL UPDATE ({t:.0f}ms)" if t else "WORLD MODEL UPDATE"
            self.sections.append((header, _render_ansi(Text("\n".join(lines2), style="blue"))))

        # -- CONTEXT --
        ctx = eng.get("context", {})
        if ctx:
            ctx_lines = [
                f"  Topic: {ctx.get('topic', 'none')}",
                f"  Turn: {ctx.get('turn', 0)}, "
                f"Recent concepts: {len(ctx.get('recent_concepts', []))}",
                f"  History turns: {ctx.get('history_length', 0)}",
            ]
            t_ctx = timings.get("context", 0)
            header = f"CONTEXT ({t_ctx:.0f}ms)" if t_ctx else "CONTEXT"
            self.sections.append((header, _render_ansi(Text("\n".join(ctx_lines), style="cyan"))))

        # -- GOALS --
        gls = eng.get("goals", {})
        if gls and gls.get("n_goals", 0) > 0:
            goal_lines = [
                f"  Total: {gls.get('n_goals', 0)}",
                f"  Active: {', '.join(gls.get('active', [])[:3]) or 'none'}",
            ]
            completed = gls.get("completed", [])
            if completed:
                goal_lines.append(f"  Completed: {', '.join(completed[:2])}")
            blocked = gls.get("blocked", [])
            if blocked:
                goal_lines.append(f"  Blocked: {', '.join(b['goal'] for b in blocked[:2])}")
            current_task = gls.get("current_task", "")
            if current_task:
                goal_lines.append(f"  Current: {current_task}")
            t_goal = timings.get("goals", 0)
            header = f"GOALS ({t_goal:.0f}ms)" if t_goal else "GOALS"
            self.sections.append((header, _render_ansi(Text("\n".join(goal_lines), style="green"))))

        # -- PLANNING --
        plan = eng.get("plan", {})
        if plan and plan.get("steps"):
            plan_lines = [f"  Steps: {plan.get('n_steps', 0)}"]
            for step in plan.get("steps", [])[:4]:
                action = step.get("action", "?")
                status = step.get("status", "pending")
                plan_lines.append(f"    [{status}] {action}")
            if plan.get("failed_steps", 0) > 0:
                plan_lines.append(f"  ⚠ {plan['failed_steps']} failed step(s) — replanning")
            t_plan = timings.get("planning", 0)
            header = f"PLANNING ({t_plan:.0f}ms)" if t_plan else "PLANNING"
            self.sections.append((header, _render_ansi(Text("\n".join(plan_lines), style="yellow"))))

        # -- SIMULATION --
        sim = eng.get("simulation", {})
        if sim and sim.get("outcomes"):
            sim_lines = []
            for outcome in sim.get("outcomes", [])[:4]:
                action = outcome.get("action", "?")
                prob = outcome.get("probability", 0)
                sim_lines.append(f"  {action}  ({prob:.2f})")
            t_sim = timings.get("simulation", 0)
            header = f"SIMULATION ({t_sim:.0f}ms)" if t_sim else "SIMULATION"
            self.sections.append((header, _render_ansi(Text("\n".join(sim_lines), style="blue"))))

        # -- REASONING --
        inf = eng.get("inference_result", {})
        chain_lines = []
        if inf:
            inferences = inf.get("inferences", [])
            contradictions = inf.get("contradictions", [])
            for inv in inferences[:4]:
                conf = inv.get("confidence", 0)
                src = inv.get("source", "?")
                pred = inv.get("predicate", "?")
                tgt = inv.get("target", "?")
                chain_lines.append(f"  {src} -[{pred}]→ {tgt}  ({conf:.2f})")
                steps = inv.get("steps", [])
                if steps:
                    chain_lines.append(f"    ∵ {steps[0].get('premise', '')}" if isinstance(steps[0], dict) else "")
            for c in contradictions[:2]:
                desc = c.get("description", "")
                chain_lines.append(f"  ⚠ {desc}")
            pt = inf.get("processing_time_ms", 0)
            header = f"REASONING ({pt:.0f}ms)" if pt else "REASONING"
            self.sections.append((header, _render_ansi(Text("\n".join(chain_lines), style="magenta"))))
        else:
            self.sections.append(("REASONING", _render_ansi(Text("No new inferences.", style="dim white"))))

        # -- LANGUAGE --
        gen = eng.get("generated_text", "")
        if gen:
            t_lang = timings.get("language", 0)
            header = f"LANGUAGE ({t_lang:.0f}ms)" if t_lang else "LANGUAGE"
            self.sections.append((header, _render_ansi(Text(gen, style="white bold"))))

        # -- LEARNING --
        resolutions = eng.get("learning_resolutions", [])
        if resolutions:
            learn_lines = []
            for r in resolutions:
                learn_lines.append(
                    f"  Resolved: {r.get('loser', '?')} "
                    f"({r.get('loser_original', 0):.2f}→{r.get('loser_new', 0):.2f})"
                )
            t_learn = timings.get("learning", 0)
            header = f"LEARNING ({t_learn:.0f}ms)" if t_learn else "LEARNING"
            self.sections.append((
                header,
                _render_ansi(Text("\n".join(learn_lines), style="dim cyan"))
            ))

        # -- MEMORY --
        mem_lines = [f"Graph: {post_c} concepts, {post_r} relations"]
        if new_c > 0 or new_r > 0:
            mem_lines.append(f"This session: +{new_c} concepts, +{new_r} relations")
        mem_lines.append(f"Last learned: \"{eng.get('text', '')[:60]}\"")
        total_t = sum(timings.values())
        header = f"MEMORY ({total_t:.0f}ms total)" if total_t else "MEMORY"
        self.sections.append((header, _render_ansi(Text("\n".join(mem_lines), style="dim white"))))


class OutputStore:
    """Stores conversation turns and produces formatted text for the output pane."""

    def __init__(self):
        self.turns: list[Turn] = []

    def add_turn(self, user_text: str, engine_result: dict[str, Any]) -> Turn:
        turn = Turn(user_text, engine_result)
        self.turns.append(turn)
        return turn

    def get_formatted_text(self):
        """Return formatted text for prompt_toolkit."""
        parts: list[str] = []
        for i, turn in enumerate(self.turns):
            if i > 0:
                parts.append("\n")
            parts.append(f">>> {turn.user_text}\n")
            for section_name, ansi_text in turn.sections:
                sep_len = max(0, 70 - len(section_name) - 8)
                parts.append(f"━━━━━━ {section_name} ━━━━━━\n")
                parts.append(ansi_text + "\n")
        return PTK_ANSI("".join(parts))
