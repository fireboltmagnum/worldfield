"""WorldField cognitive pipeline TUI — prompt_toolkit Application."""
from __future__ import annotations

import asyncio
import threading

from prompt_toolkit import Application, get_app
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
    """Manages the prompt_toolkit Application lifecycle."""

    def __init__(self,
                 cfg: Config | None = None,
                 engine: Engine | None = None,
                 reasoner: ReasoningEngine | None = None):
        self.cfg = cfg
        self.engine = engine
        self.reasoner = reasoner
        self.store = OutputStore()
        self.session_additions = [0, 0]  # [concepts, relations]
        self._processing = False

        # Pre-load slow models in background so first interaction is faster
        if self.engine is not None:
            threading.Thread(target=self._preload_models, daemon=True).start()

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
        async def _enter(event):
            buffer = event.current_buffer
            text = buffer.text.strip()
            if not text or self._processing:
                return
            buffer.reset()

            if text.startswith("/"):
                self._handle_command(text)
                self.output_control.invalidate()
                self.header_control.invalidate()
            else:
                app = get_app()
                app.create_background_task(self._process_async(text))

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

    def _preload_models(self):
        """Eagerly load slow models so first input doesn't block."""
        try:
            _ = self.engine.text_encoder
            _ = self.engine.nlp
        except Exception:
            pass

    def _get_header_text(self):
        return ANSI(render_header(self.engine, tuple(self.session_additions)))

    def _get_output_text(self):
        return self.store.get_formatted_text()

    def _placeholder_turn_dict(self, text: str,
                                label: str = "") -> dict:
        t = self.engine.graph.n_concepts
        r = self.engine.graph.n_relations
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

    async def _process_async(self, text: str):
        """Run engine processing in a thread pool (non-blocking)."""
        self._processing = True
        loop = asyncio.get_event_loop()

        # Show processing indicator immediately
        self.store.add_turn(text, self._placeholder_turn_dict(
            text, "Processing..."
        ))
        self.output_control.invalidate()

        try:
            result = await loop.run_in_executor(
                None, self.engine.process, text
            )
            answer = None
            if self.reasoner is not None:
                answer = await loop.run_in_executor(
                    None, self.reasoner.answer, text
                )
            result["_answer"] = answer

            pre_c, pre_r = result.get("graph_pre_state", (0, 0))
            self.session_additions[0] += result.get(
                "total_concepts", 0
            ) - pre_c
            self.session_additions[1] += result.get(
                "total_relations", 0
            ) - pre_r

            new_turn = Turn(text, result)
        except Exception as e:
            err = self._placeholder_turn_dict(text, f"Error: {e}")
            new_turn = Turn(text, err)

        # Replace placeholder with real result
        self.store.turns[-1] = new_turn
        self._processing = False
        self.output_control.invalidate()
        self.header_control.invalidate()

    def _on_submit(self, buffer: Buffer) -> bool:
        # Only used for commands; free text handled by async _enter
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
            self.store.add_turn(text, self._placeholder_turn_dict(
                text, f"Importing {path}..."
            ))
            self.output_control.invalidate()
            try:
                from PIL import Image
                img = Image.open(path)
                result = self.engine.process_image(img, source=path)
                result["_answer"] = None
                self.store.turns[-1] = Turn(f"/import {path}", result)
            except Exception as e:
                self.store.turns[-1] = Turn(
                    f"/import {path}",
                    self._placeholder_turn_dict(text, f"Import error: {e}"),
                )
            self.output_control.invalidate()

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

    def run(self):
        self.app.run()


def run_cli():
    """Entry point — called from __main__.py."""
    # Defer heavy imports so starting the CLI is fast
    from ..config import Config
    from ..core.engine import Engine
    from ..reasoning import ReasoningEngine

    cfg = Config()
    engine = Engine(cfg)
    reasoner = ReasoningEngine(engine.graph)
    wf = WorldFieldApp(cfg, engine, reasoner)
    wf.run()
