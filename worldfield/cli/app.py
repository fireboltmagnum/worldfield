"""WorldField interactive CLI."""
from __future__ import annotations

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.align import Align

from ..core.engine import Engine
from ..config import Config
from ..reasoning import ReasoningEngine, format_answer

console = Console()


def make_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="input", size=3),
    )
    return layout


def header_panel(cfg, engine):
    g = engine.graph
    text = Text()
    text.append("WorldField ", style="bold cyan")
    text.append(f"concepts={g.n_concepts} ", style="green")
    text.append(f"relations={g.n_relations} ", style="yellow")
    text.append(f"avg_conf={g.avg_confidence:.2f}", style="dim")
    return Panel(text, style="cyan")


def chat_panel(messages):
    items = []
    for msg in messages[-20:]:
        role = msg.get("role", "")
        text = msg.get("text", "")
        if role == "user":
            items.append(Text(f">>> {text}", style="green"))
        elif role == "system":
            items.append(Text(f"[{msg.get('label', 'sys')}] {text}", style="yellow"))
        elif role == "assistant":
            for i, line in enumerate(text.split("\n")):
                items.append(Text(f"  {line}", style="white" if i == 0 else "dim"))
    if not items:
        items.append(Text("  /ask <question> to query the graph.", style="dim white"))
    return Panel(Group(*items), title="Chat", border_style="green")


def build_display(messages, engine, layout):
    cfg = Config()
    layout["header"].update(header_panel(cfg, engine))
    layout["main"].update(chat_panel(messages))
    layout["input"].update(Panel(
        Align(Text("/ask <q>  /stats  /quit", style="dim"), align="left"),
        style="white",
    ))


def run_cli():
    cfg = Config()
    engine = Engine(cfg)
    reasoner = ReasoningEngine(engine.graph)
    messages = []
    layout = make_layout()

    print()  # spacing before Live starts

    with Live(layout, refresh_per_second=4, screen=True) as live:
        build_display(messages, engine, layout)
        live.refresh()

        while True:
            try:
                user_input = console.input("[bold green]>>> [/bold green]")
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input.strip():
                continue

            messages.append({"role": "user", "text": user_input})

            if user_input.startswith("/"):
                cmd = user_input[1:].strip().lower()

                if cmd in ("quit", "exit"):
                    break
                elif cmd == "help":
                    messages.append({
                        "role": "system", "label": "help",
                        "text": "/ask <q>   Ask a question\n/stats     Graph stats\n/save      Save state\n/reset     Reset\n/quit      Exit",
                    })
                elif cmd == "stats":
                    g = engine.graph
                    msg = f"Concepts: {g.n_concepts}\nRelations: {g.n_relations}\nAvg conf: {g.avg_confidence:.3f}"
                    if g.n_concepts > 0:
                        msg += "\nTop:\n" + "\n".join(f"  {n}: {c:.3f}" for n, c in g.top_concepts(5))
                    messages.append({"role": "system", "label": "stats", "text": msg})
                elif cmd.startswith("ask "):
                    answer = reasoner.answer(cmd[4:])
                    messages.append({"role": "assistant", "text": format_answer(answer)})
                elif cmd == "save":
                    engine._save_state()
                    messages.append({"role": "system", "label": "save", "text": "Saved."})
                elif cmd == "reset":
                    engine.reset()
                    reasoner = ReasoningEngine(engine.graph)
                    messages.append({"role": "system", "label": "reset", "text": "Reset."})
                else:
                    messages.append({"role": "system", "label": "error", "text": f"Unknown: {cmd}"})
            else:
                answer = reasoner.answer(user_input)
                messages.append({"role": "assistant", "text": format_answer(answer)})

            build_display(messages, engine, layout)
            live.refresh()

    console.print("[yellow]WorldField closed.[/yellow]")
