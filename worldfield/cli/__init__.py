"""WorldField CLI — interactive chat + dashboard using rich + prompt_toolkit."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from threading import Thread, Event

from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from ..core.engine import Engine
from ..config import Config


console = Console()


def make_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="input", size=3),
    )
    layout["main"].split_row(
        Layout(name="chat", ratio=2),
        Layout(name="dashboard", ratio=1),
    )
    return layout


def header_panel(cfg: Config, engine: Engine) -> Panel:
    text = Text()
    text.append("WorldField ", style="bold cyan")
    text.append(f"| latent_dim={cfg.latent_dim}")
    text.append(f" | slots={engine.slots.active_count()}/{cfg.n_slots}")
    text.append(f" | fragments={engine.store.count}")
    if engine.graph.has_edges:
        text.append(f" | graph_edges={engine.graph.n_edges}")
    return Panel(text, style="cyan")


def chat_panel(messages: list) -> Panel:
    items = []
    for msg in messages[-12:]:
        if msg["role"] == "user":
            items.append(Text(f">>> {msg['text']}", style="green"))
        elif msg["role"] == "system":
            items.append(Text(f"[{msg.get('label', 'sys')}] {msg['text']}", style="yellow"))
        else:
            items.append(Text(f"  {msg['text']}", style="white"))
    if not items:
        items.append(Text("  No interactions yet. Type a message below.", style="dim white"))
    return Panel(Group(*items), title="Chat", border_style="green")


def dashboard_panel(engine: Engine) -> Panel:
    parts = []

    # Slots
    slot_table = Table(show_header=False, box=None, padding=(0, 1))
    slot_table.add_column("Slot", style="dim")
    slot_table.add_column("Dim", style="cyan")
    active = engine.slots.active_slots()
    for i, vec in enumerate(active):
        mag = float(np.linalg.norm(vec))
        slot_table.add_row(f"Slot {i}", f"{mag:.3f}")
    if len(active) == 0:
        slot_table.add_row("(empty)", "")
    parts.append(Panel(slot_table, title="Slots", border_style="blue"))

    # Graph info
    if engine.graph.has_edges:
        g_text = Text()
        g_text.append(f"Edges: {engine.graph.n_edges}\n")
        g_text.append(f"Events: {engine.graph.N}")
        parts.append(Panel(g_text, title="Graph", border_style="magenta"))
    else:
        parts.append(Panel(Text("(no edges yet)", style="dim"), title="Graph", border_style="magenta"))

    return Panel(Group(*parts), title="Dashboard", border_style="blue")


def build_display(messages, engine, cfg, layout):
    layout["header"].update(header_panel(cfg, engine))
    layout["chat"].update(chat_panel(messages))
    layout["dashboard"].update(dashboard_panel(engine))
    layout["input"].update(Panel(Align(Text("Type a message and press Enter. Commands: /help, /reset, /stats", style="dim"), align="left"), style="white"))


def run_cli():
    cfg = Config()
    engine = Engine(cfg)
    messages = []
    layout = make_layout()

    print("Initializing WorldField...")

    with Live(layout, refresh_per_second=4, screen=True) as live:
        build_display(messages, engine, cfg, layout)
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
                if cmd == "quit" or cmd == "exit":
                    break
                elif cmd == "reset":
                    engine.reset()
                    messages.append({"role": "system", "label": "reset", "text": "Engine state reset."})
                    build_display(messages, engine, cfg, layout)
                    live.refresh()
                    continue
                elif cmd == "stats":
                    msg = (
                        f"Slots: {engine.slots.active_count()}/{cfg.n_slots} | "
                        f"Fragments: {engine.store.count} | "
                        f"Graph edges: {engine.graph.n_edges if engine.graph else 0}"
                    )
                    messages.append({"role": "system", "label": "stats", "text": msg})
                    build_display(messages, engine, cfg, layout)
                    live.refresh()
                    continue
                elif cmd == "help":
                    help_text = (
                        "Commands:\n"
                        "  /help        Show this help\n"
                        "  /reset       Reset engine state (keeps fragments)\n"
                        "  /stats       Show system stats\n"
                        "  /query <t>   Query without storing\n"
                        "  /refine <t>  Run iterative refinement\n"
                        "  /save        Force save state\n"
                        "  /quit        Exit"
                    )
                    messages.append({"role": "system", "label": "help", "text": help_text})
                    build_display(messages, engine, cfg, layout)
                    live.refresh()
                    continue
                elif cmd.startswith("query "):
                    query_text = cmd[6:]
                    result = engine.query(query_text)
                    n = len(result.get("related", []))
                    reasoning = result.get("reasoning", {})
                    msg = f"Found {n} related fragments"
                    if reasoning:
                        msg += f" | {reasoning.get('n_edges', 0)} edges traversed"
                    messages.append({"role": "system", "label": "query", "text": msg})
                    build_display(messages, engine, cfg, layout)
                    live.refresh()
                    continue
                elif cmd.startswith("refine "):
                    ref_text = cmd[7:]
                    traj = engine.refine(ref_text)
                    if traj:
                        final = traj[-1]
                        msg = (
                            f"Refinement: {len(traj)} iters, "
                            f"final active={final['active_nodes']}, "
                            f"top_score={final['top_score']:.3f}"
                        )
                    else:
                        msg = "Refinement: no trajectory (no edges or no matches)"
                    messages.append({"role": "system", "label": "refine", "text": msg})
                    build_display(messages, engine, cfg, layout)
                    live.refresh()
                    continue
                elif cmd == "save":
                    engine._save_state()
                    messages.append({"role": "system", "label": "save", "text": "State saved."})
                    build_display(messages, engine, cfg, layout)
                    live.refresh()
                    continue
                else:
                    messages.append({"role": "system", "label": "error", "text": f"Unknown command: {cmd}"})
                    build_display(messages, engine, cfg, layout)
                    live.refresh()
                    continue

            # Process as text input
            try:
                result = engine.process(user_input)
                related = result.get("related", [])
                n_rel = len(related)
                top = related[0] if related else None
                msg_parts = [f"Stored fragment. {n_rel} related."]
                if top:
                    msg_parts.append(f"Best match: {top['score']:.3f}")
                if result.get("reasoning"):
                    msg_parts.append(f"Graph: {result['reasoning'].get('n_edges', 0)} edges")
                messages.append({"role": "assistant", "text": " | ".join(msg_parts)})
            except Exception as e:
                messages.append({"role": "system", "label": "error", "text": f"Error: {e}"})

            build_display(messages, engine, cfg, layout)
            live.refresh()

    console.print("[yellow]WorldField closed.[/yellow]")
