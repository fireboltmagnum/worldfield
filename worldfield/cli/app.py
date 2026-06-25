"""WorldField cognitive pipeline TUI — prompt_toolkit Application."""
from __future__ import annotations

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import ANSI

from ..core.engine import Engine
from ..config import Config
from ..reasoning import ReasoningEngine
from .chat import OutputStore
from .dashboard import render_header

_command_completer = WordCompleter([
    "/import ", "/stats", "/save", "/reset", "/help", "/quit",
])

_style = Style([
    ("output", "bg:ansiblack"),
    ("input", "bg:ansiblack fg:ansiwhite"),
    ("header", "bg:ansiblack"),
])


class WorldFieldApp:
    """Manages the prompt_toolkit Application lifecycle."""

    def __init__(self):
        self.cfg = Config()
        self.engine = Engine(self.cfg)
        self.reasoner = ReasoningEngine(self.engine.graph)
        self.store = OutputStore()
        self.session_additions = [0, 0]  # [concepts, relations]

        # -- Input buffer --
        self.input_buffer = Buffer(
            history=FileHistory(".worldfield_history"),
            completer=_command_completer,
            accept_handler=self._on_submit,
        )

        # -- Output display --
        self.output_control = FormattedTextControl(
            text=self._get_output_text,
            show_cursor=False,
        )

        # -- Header --
        self.header_control = FormattedTextControl(
            text=self._get_header_text,
            show_cursor=False,
        )

        # -- Layout --
        self.layout = Layout(
            HSplit([
                Window(content=self.header_control, height=2, dont_extend_height=True, style="class:header"),
                Window(content=self.output_control, wrap_lines=True, style="class:output"),
                Window(content=BufferControl(buffer=self.input_buffer), height=1, dont_extend_height=True, style="class:input"),
            ])
        )

        # -- Key bindings --
        kb = KeyBindings()

        @kb.add("enter")
        def _enter(event):
            event.current_buffer.validate_and_handle()

        @kb.add("c-c")
        def _ctrl_c(event):
            event.app.exit()

        @kb.add("c-d")
        def _ctrl_d(event):
            event.app.exit()

        self.kb = kb

        # -- App --
        self.app = Application(
            layout=self.layout,
            key_bindings=self.kb,
            style=_style,
            full_screen=True,
            mouse_support=True,
        )

    def _get_header_text(self):
        return ANSI(render_header(self.engine, tuple(self.session_additions)))

    def _get_output_text(self):
        return self.store.get_formatted_text()

    def _on_submit(self, buffer: Buffer) -> bool:
        text = buffer.text.strip()
        if not text:
            return False
        buffer.reset()

        if text.startswith("/"):
            self._handle_command(text)
        else:
            self._handle_input(text)

        self.output_control.invalidate()
        self.header_control.invalidate()
        return True

    def _handle_command(self, text: str):
        cmd = text[1:].strip().lower()

        if cmd in ("quit", "exit"):
            self.app.exit()

        elif cmd == "help":
            help_text = (
                "Commands:\n"
                "  free text    Process through cognitive pipeline\n"
                "  /import <p>  Process image file\n"
                "  /stats       Graph statistics\n"
                "  /save        Persist graph state\n"
                "  /reset       Clear engine state\n"
                "  /help        This message\n"
                "  /quit        Exit"
            )
            self.store.add_turn(text, {
                "text": help_text,
                "concepts_extracted": [],
                "extracted_concepts_raw": [],
                "extracted_relations_raw": [],
                "graph_query": {},
                "timings": {},
                "total_concepts": self.engine.graph.n_concepts,
                "total_relations": self.engine.graph.n_relations,
                "graph_pre_state": (self.engine.graph.n_concepts, self.engine.graph.n_relations),
            })

        elif cmd == "stats":
            g = self.engine.graph
            info = f"Concepts: {g.n_concepts}\nRelations: {g.n_relations}\nAvg conf: {g.avg_confidence:.3f}"
            if g.n_concepts > 0:
                info += "\nTop:\n" + "\n".join(f"  {n}: {c:.3f}" for n, c in g.top_concepts(5))
            self.store.add_turn(text, {
                "text": info,
                "concepts_extracted": [],
                "extracted_concepts_raw": [],
                "extracted_relations_raw": [],
                "graph_query": {},
                "timings": {},
                "total_concepts": g.n_concepts,
                "total_relations": g.n_relations,
                "graph_pre_state": (g.n_concepts, g.n_relations),
            })

        elif cmd.startswith("import "):
            path = cmd[7:].strip()
            self._handle_import(path)

        elif cmd == "save":
            self.engine._save_state()
            self.store.add_turn(text, {
                "text": "State saved.",
                "concepts_extracted": [],
                "extracted_concepts_raw": [],
                "extracted_relations_raw": [],
                "graph_query": {},
                "timings": {},
                "total_concepts": self.engine.graph.n_concepts,
                "total_relations": self.engine.graph.n_relations,
                "graph_pre_state": (self.engine.graph.n_concepts, self.engine.graph.n_relations),
            })

        elif cmd == "reset":
            self.engine.reset()
            self.session_additions = [0, 0]
            self.store.add_turn(text, {
                "text": "Engine state reset.",
                "concepts_extracted": [],
                "extracted_concepts_raw": [],
                "extracted_relations_raw": [],
                "graph_query": {},
                "timings": {},
                "total_concepts": self.engine.graph.n_concepts,
                "total_relations": self.engine.graph.n_relations,
                "graph_pre_state": (self.engine.graph.n_concepts, self.engine.graph.n_relations),
            })

        else:
            self.store.add_turn(text, {
                "text": f"Unknown command: {cmd}. Type /help for commands.",
                "concepts_extracted": [],
                "extracted_concepts_raw": [],
                "extracted_relations_raw": [],
                "graph_query": {},
                "timings": {},
                "total_concepts": self.engine.graph.n_concepts,
                "total_relations": self.engine.graph.n_relations,
                "graph_pre_state": (self.engine.graph.n_concepts, self.engine.graph.n_relations),
            })

    def _handle_input(self, text: str):
        try:
            result = self.engine.process(text)
            answer = self.reasoner.answer(text)
            result["_answer"] = answer
            self.store.add_turn(text, result)
            # Track session growth
            pre_c = result.get("graph_pre_state", (0, 0))[0]
            pre_r = result.get("graph_pre_state", (0, 0))[1]
            self.session_additions[0] += self.engine.graph.n_concepts - pre_c
            self.session_additions[1] += self.engine.graph.n_relations - pre_r
        except Exception as e:
            self.store.add_turn(text, {
                "text": f"Error: {e}",
                "concepts_extracted": [],
                "extracted_concepts_raw": [],
                "extracted_relations_raw": [],
                "graph_query": {},
                "timings": {},
                "total_concepts": self.engine.graph.n_concepts,
                "total_relations": self.engine.graph.n_relations,
                "graph_pre_state": (self.engine.graph.n_concepts, self.engine.graph.n_relations),
            })

    def _handle_import(self, path: str):
        try:
            from PIL import Image
            img = Image.open(path)
            result = self.engine.process_image(img, source=path)
            result["_answer"] = None
            self.store.add_turn(f"/import {path}", result)
        except Exception as e:
            self.store.add_turn(f"/import {path}", {
                "text": f"Import error: {e}",
                "concepts_extracted": [],
                "extracted_concepts_raw": [],
                "extracted_relations_raw": [],
                "graph_query": {},
                "timings": {},
                "total_concepts": self.engine.graph.n_concepts,
                "total_relations": self.engine.graph.n_relations,
                "graph_pre_state": (self.engine.graph.n_concepts, self.engine.graph.n_relations),
            })

    def run(self):
        self.app.run()


def run_cli():
    """Entry point — called from __main__.py."""
    wf = WorldFieldApp()
    wf.run()
