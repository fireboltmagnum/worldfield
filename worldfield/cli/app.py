"""WorldField cognitive pipeline TUI — prompt_toolkit Application."""
from __future__ import annotations

import asyncio
import threading

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import ANSI

from .chat import OutputStore, Turn
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
    def __init__(self, cfg=None, engine=None, reasoner=None):
        self.cfg = cfg
        self.engine = engine
        self.reasoner = reasoner
        self.store = OutputStore()
        self.session_additions = [0, 0]
        self._processing = False

        if self.engine is not None:
            threading.Thread(target=self._preload_models, daemon=True).start()

        self.input_buffer = Buffer(
            history=FileHistory(".worldfield_history"),
            completer=_command_completer,
            accept_handler=self._on_submit,
        )

        self.output_control = FormattedTextControl(
            text=self._get_output_text,
            show_cursor=False,
        )

        self.header_control = FormattedTextControl(
            text=self._get_header_text,
            show_cursor=False,
        )

        self.layout = Layout(
            HSplit([
                Window(content=self.header_control, height=2, dont_extend_height=True, style="class:header"),
                Window(content=self.output_control, wrap_lines=True, style="class:output"),
                Window(content=BufferControl(buffer=self.input_buffer), height=1, dont_extend_height=True, style="class:input"),
            ])
        )

        kb = KeyBindings()

        @kb.add("enter")
        async def _enter(event):
            buffer = event.current_buffer
            text = buffer.text.strip()
            if not text or self._processing:
                return
            buffer.reset()
            if text.startswith("/"):
                self._handle_command(text)
            else:
                event.app.create_background_task(self._process_async(text))

        @kb.add("c-c")
        def _ctrl_c(event):
            event.app.exit()

        @kb.add("c-d")
        def _ctrl_d(event):
            event.app.exit()

        self.kb = kb

        self.app = Application(
            layout=self.layout,
            key_bindings=self.kb,
            style=_style,
            full_screen=True,
            mouse_support=True,
        )

    # ── Helpers ────────────────────────────────────────────────────

    def _refresh(self):
        self.app.invalidate()

    def _graph_stats(self):
        if self.engine and self.engine.graph:
            g = self.engine.graph
            return g.n_concepts, g.n_relations
        return 0, 0

    def _info_turn(self, text: str, body: str) -> dict:
        t, r = self._graph_stats()
        return {
            "text": body,
            "concepts_extracted": [],
            "extracted_concepts_raw": [],
            "extracted_relations_raw": [],
            "graph_query": {},
            "timings": {},
            "total_concepts": t,
            "total_relations": r,
            "graph_pre_state": (t, r),
        }

    def _placeholder_turn(self, text: str, label: str = "") -> dict:
        t, r = self._graph_stats()
        return {
            "text": label or text,
            "concepts_extracted": [],
            "extracted_concepts_raw": [],
            "extracted_relations_raw": [],
            "graph_query": {},
            "timings": {},
            "total_concepts": t,
            "total_relations": r,
            "graph_pre_state": (t, r),
        }

    # ── Models ─────────────────────────────────────────────────────

    def _preload_models(self):
        try:
            if self.engine:
                _ = self.engine.text_encoder
                _ = self.engine.nlp
        except Exception:
            pass

    def _get_header_text(self):
        return ANSI(render_header(self.engine, tuple(self.session_additions)))

    def _get_output_text(self):
        return self.store.get_formatted_text()

    # ── Processing ─────────────────────────────────────────────────

    async def _process_async(self, text: str):
        loop = asyncio.get_event_loop()
        self._processing = True

        placeholder = self._placeholder_turn(text, "Processing...")
        self.store.add_turn(text, placeholder)
        self._refresh()

        try:
            result = await loop.run_in_executor(None, self.engine.process, text)
            answer = None
            if self.reasoner is not None:
                try:
                    answer = await loop.run_in_executor(None, self.reasoner.answer, text)
                except Exception:
                    pass
            result["_answer"] = answer

            pre_c, pre_r = result.get("graph_pre_state", (0, 0))
            self.session_additions[0] += result.get("total_concepts", 0) - pre_c
            self.session_additions[1] += result.get("total_relations", 0) - pre_r

            new_turn = Turn(text, result)
        except Exception as e:
            err_body = f"Pipeline error: {e}"
            new_turn = Turn(text, self._placeholder_turn(text, err_body))
        finally:
            self.store.turns[-1] = new_turn
            self._processing = False
            self._refresh()

    def _on_submit(self, buffer: Buffer) -> bool:
        return True

    def _handle_command(self, text: str):
        cmd = text[1:].strip().lower()

        if cmd in ("quit", "exit"):
            self.app.exit()
            return

        if cmd == "help":
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
            self.store.add_turn(text, self._info_turn(text, help_text))
            self._refresh()
            return

        if cmd == "stats":
            g = self.engine.graph if self.engine else None
            if g and g.n_concepts > 0:
                info = f"Concepts: {g.n_concepts}\nRelations: {g.n_relations}\nAvg conf: {g.avg_confidence:.3f}"
                info += "\nTop:\n" + "\n".join(f"  {n}: {c:.3f}" for n, c in g.top_concepts(5))
            else:
                info = "Graph is empty. Type something to build knowledge."
            self.store.add_turn(text, self._info_turn(text, info))
            self._refresh()
            return

        if cmd.startswith("import "):
            path = cmd[7:].strip()
            self.store.add_turn(text, self._placeholder_turn(text, f"Importing {path}..."))
            self._refresh()
            try:
                from PIL import Image
                img = Image.open(path)
                result = self.engine.process_image(img, source=path)
                result["_answer"] = None
                self.store.turns[-1] = Turn(f"/import {path}", result)
            except Exception as e:
                self.store.turns[-1] = Turn(
                    f"/import {path}",
                    self._placeholder_turn(text, f"Import error: {e}"),
                )
            self._refresh()
            return

        if cmd == "save":
            if self.engine:
                self.engine._save_state()
                self.store.add_turn(text, self._info_turn(text, "State saved."))
            else:
                self.store.add_turn(text, self._info_turn(text, "No engine to save."))
            self._refresh()
            return

        if cmd == "reset":
            if self.engine:
                self.engine.reset()
            self.session_additions = [0, 0]
            self.store.add_turn(text, self._info_turn(text, "Engine state reset."))
            self._refresh()
            return

        self.store.add_turn(
            text,
            self._info_turn(text, f"Unknown command: {cmd}. Type /help for commands."),
        )
        self._refresh()

    def run(self):
        self.app.run()


def run_cli():
    """Entry point — called from __main__.py."""
    import sys

    def _status(msg):
        sys.stdout.write("\r\x1b[K  " + msg)
        sys.stdout.flush()

    _status("Importing packages...")
    from ..config import Config
    from ..core.engine import Engine
    from ..reasoning import ReasoningEngine

    cfg = Config()

    _status("Building cognitive pipeline...")
    engine = Engine(cfg)

    _status("Loading reasoning engine...")
    reasoner = ReasoningEngine(engine.graph)

    wf = WorldFieldApp(cfg, engine, reasoner)
    wf.run()
