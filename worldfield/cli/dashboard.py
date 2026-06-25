"""CLI dashboard header — graph stats bar."""
from __future__ import annotations

from rich.console import Console
from rich.text import Text

_rich_console = Console(color_system="truecolor", force_terminal=True, width=80)


def render_header(engine, session_additions: tuple[int, int] = (0, 0)) -> str:
    """Render the top header bar as an ANSI string."""
    g = engine.graph
    add_c, add_r = session_additions
    text = Text()
    text.append("WorldField", style="bold cyan")
    text.append(f"  concepts={g.n_concepts}", style="green")
    text.append(f"  relations={g.n_relations}", style="yellow")
    text.append(f"  conf={g.avg_confidence:.2f}", style="dim")
    if add_c or add_r:
        text.append(f"  session=+{add_c}c +{add_r}r", style="blue")
    text.append("\n" + "\u2500" * 78, style="dim")
    with _rich_console.capture() as cap:
        _rich_console.print(text)
    return cap.get()
