"""CLI dashboard header — graph stats bar (plain text, no rich)."""


def render_header(engine, session_additions: tuple[int, int] = (0, 0)) -> str:
    g = engine.graph
    add_c, add_r = session_additions
    parts = [
        f"WorldField  concepts={g.n_concepts}  relations={g.n_relations}  conf={g.avg_confidence:.2f}",
    ]
    if add_c or add_r:
        parts[0] += f"  session=+{add_c}c +{add_r}r"
    parts.append("\u2500" * 78)
    return "\n".join(parts)
