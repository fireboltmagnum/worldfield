"""CLI output store and section rendering for the cognitive pipeline display.

Plain-text only — no rich, no ANSI formatting. Sections are separated
with Unicode box-drawing characters for clean terminal output.
"""
from __future__ import annotations

from typing import Any


class Turn:
    """One user input + engine response, rendered into sections."""

    def __init__(self, user_text: str, engine_result: dict[str, Any]):
        self.user_text = user_text
        self.engine = engine_result
        self.sections: list[tuple[str, str]] = []
        self._build()

    def _build(self):
        eng = self.engine
        timings = eng.get("timings", {})

        # -- INPUT --
        self.sections.append(("INPUT", eng.get("text", "")))

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
            self.sections.append((header, "\n".join(lines)))

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
            self.sections.append((header, "\n".join(rows)))

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
                lines_ws.append(f"  competing interpretations: {ws['n_alternatives']}")
            t = timings.get("world_state", 0)
            header = f"WORLD STATE ({t:.0f}ms)" if t else "WORLD STATE"
            self.sections.append((header, "\n".join(lines_ws)))

        # -- CONTEXT --
        cw = eng.get("context_window")
        if cw:
            ctx_lines = []
            if cw.get("topic_stack"):
                topics = " > ".join(cw["topic_stack"])
                ctx_lines.append(f"  topics: {topics}")
            entities = cw.get("entities", [])
            if entities:
                ent_str = ", ".join(f"{e['name']}({e['mention_count']})" for e in entities[:8])
                ctx_lines.append(f"  entities: {ent_str}")
            ctx_lines.append(f"  turn: {cw.get('turn', '?')}  events: {cw.get('n_events', 0)}  "
                  f"world_states: {cw.get('n_world_states', 0)}")
            refs = cw.get("unresolved_refs", [])
            if refs:
                ctx_lines.append(f"  unresolved: {', '.join(r['surface'] for r in refs)}")
            t_cw = timings.get("context_window", 0)
            header = f"CONTEXT ({t_cw:.0f}ms)" if t_cw else "CONTEXT"
            self.sections.append((header, "\n".join(ctx_lines)))
        else:
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
                self.sections.append((header, "\n".join(ctx_lines)))

        # -- ATTENTION --
        attn = eng.get("attention")
        if attn:
            attn_lines = []
            if attn.task_mode != "browsing":
                attn_lines.append(f"  Mode: {attn.task_mode}")
            for sc in attn.attended[:10]:
                bar = "#" * int(sc.score * 20) + "-" * (20 - int(sc.score * 20))
                attn_lines.append(f"  + {sc.name:<20} {sc.score:.3f} {bar}")
            if attn.suppressed:
                suppressed_str = ", ".join(f"{s.name}({s.score:.2f})" for s in attn.suppressed[:5])
                attn_lines.append(f"  suppressed: {suppressed_str}")
            if attn.n_candidates:
                attn_lines.append(f"  candidates: {attn.n_candidates}")
            t_attn = timings.get("attention", 0)
            header = f"ATTENTION ({t_attn:.0f}ms)" if t_attn else "ATTENTION"
            self.sections.append((header, "\n".join(attn_lines)))

        # -- RETRIEVAL --
        mr = eng.get("memory_retrieval")
        if mr:
            mr_lines = []
            node_names = list(mr.nodes.keys())
            mr_lines.append(f"  nodes ({len(node_names)}): {', '.join(node_names[:10])}")
            mr_lines.append(f"  edges: {len(mr.edges)}")
            if mr.pruned > 0:
                mr_lines.append(f"  pruned: {mr.pruned}")
            t_mr = timings.get("memory_retrieval", 0)
            header = f"RETRIEVAL ({t_mr:.0f}ms)" if t_mr else "RETRIEVAL"
            self.sections.append((header, "\n".join(mr_lines)))

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
            self.sections.append((header, "\n".join(goal_lines)))

        # -- PLANNING --
        plan = eng.get("plan", {})
        steps = plan.get("latest_plan") if isinstance(plan, dict) else None
        if steps:
            plan_lines = [f"  Plans created: {plan.get('n_plans', 1)}"]
            for step in steps[:4]:
                action = step.get("action", "?")
                status = step.get("status", "pending")
                plan_lines.append(f"    [{status}] {action}")
            t_plan = timings.get("planning", 0)
            header = f"PLANNING ({t_plan:.0f}ms)" if t_plan else "PLANNING"
            self.sections.append((header, "\n".join(plan_lines)))

        # -- SIMULATION --
        sim_raw = eng.get("simulation", [])
        if isinstance(sim_raw, list) and sim_raw:
            sim_lines = []
            for outcome in sim_raw[:4]:
                desc = outcome.get("description", outcome.get("action", "?"))
                prob = outcome.get("probability", 0)
                sim_lines.append(f"  {desc}  ({prob:.2f})")
            t_sim = timings.get("simulation", 0)
            header = f"SIMULATION ({t_sim:.0f}ms)" if t_sim else "SIMULATION"
            self.sections.append((header, "\n".join(sim_lines)))

        # -- WORLD MODEL UPDATE --
        pre_c, pre_r = eng.get("graph_pre_state", (0, 0))
        post_c = eng.get("total_concepts", 0)
        post_r = eng.get("total_relations", 0)
        new_c = post_c - pre_c
        new_r = post_r - pre_r
        rels_raw = eng.get("extracted_relations_raw", [])
        if new_c > 0 or new_r > 0 or rels_raw:
            lines2 = []
            for r in rels_raw[:5]:
                src = r.get("source", "?")
                pred = r.get("predicate", "?")
                tgt = r.get("target", "?")
                lines2.append(f"  {src} -[{pred}]? {tgt}   (new)")
            if not lines2 and (new_c > 0 or new_r > 0):
                lines2.append(f"  +{new_c} concepts, +{new_r} relations stored")
            t = timings.get("world_update", 0)
            header = f"WORLD MODEL UPDATE ({t:.0f}ms)" if t else "WORLD MODEL UPDATE"
            self.sections.append((header, "\n".join(lines2)))

        # -- REASONING --
        inf = eng.get("inference_result", {})
        if inf:
            chain_lines = []
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
                    step0 = steps[0]
                    premise = step0.get("premise", "") if isinstance(step0, dict) else ""
                    if premise:
                        chain_lines.append(f"    \u2235 {premise}")
            for c in contradictions[:2]:
                desc = c.get("description", "")
                chain_lines.append(f"  \u26a0 {desc}")
            pt = inf.get("processing_time_ms", 0)
            header = f"REASONING ({pt:.0f}ms)" if pt else "REASONING"
            self.sections.append((header, "\n".join(chain_lines)))
        else:
            self.sections.append(("REASONING", "No new inferences."))

        # -- LANGUAGE --
        gen = eng.get("generated_text", "")
        if gen:
            t_lang = timings.get("language", 0)
            header = f"LANGUAGE ({t_lang:.0f}ms)" if t_lang else "LANGUAGE"
            self.sections.append((header, gen))

        # -- MEMORY --
        mem_lines = [f"Graph: {post_c} concepts, {post_r} relations"]
        if new_c > 0 or new_r > 0:
            mem_lines.append(f"This session: +{new_c} concepts, +{new_r} relations")
        mem_lines.append(f"Last learned: \"{eng.get('text', '')[:60]}\"")
        total_t = sum(timings.values()) if timings else 0
        header = f"MEMORY ({total_t:.0f}ms total)" if total_t else "MEMORY"
        self.sections.append((header, "\n".join(mem_lines)))

        # -- LEARNING --
        resolutions = eng.get("learning_resolutions", [])
        if resolutions:
            learn_lines = []
            for r in resolutions:
                learn_lines.append(
                    f"  Resolved: {r.get('loser', '?')} "
                    f"({r.get('loser_original', 0):.2f}\u2192{r.get('loser_new', 0):.2f})"
                )
            t_learn = timings.get("learning", 0)
            header = f"LEARNING ({t_learn:.0f}ms)" if t_learn else "LEARNING"
            self.sections.append((header, "\n".join(learn_lines)))


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
            for section_name, section_text in turn.sections:
                sep_len = max(0, 70 - len(section_name) - 8)
                parts.append(f"\u2501\u2501\u2501\u2501 {section_name} \u2501\u2501\u2501\u2501\n")
                parts.append(section_text + "\n")
        return "".join(parts)
