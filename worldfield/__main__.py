"""WorldField CLI entry point: python -m worldfield [command]

Commands:
  (no args)   Launch interactive CLI
  --check-deps  Check/install dependencies first
  reason      Ask a natural language question
  process     Process a single text input
  query       Query without storing
  stats       Show graph statistics
"""
from __future__ import annotations

import sys

from .launcher import check_deps
from .config import Config
from .core.engine import Engine
from .reasoning import ReasoningEngine, format_answer


def main():
    # Auto-install dependencies on first run
    if "--check-deps" in sys.argv:
        check_deps()
        sys.argv.remove("--check-deps")

    args = sys.argv[1:]

    if not args:
        from .cli.app import run_cli
        run_cli()
        return

    cfg = Config()
    cmd = args[0]

    if cmd == "reason":
        text = " ".join(args[1:])
        engine = Engine(cfg)
        reasoner = ReasoningEngine(engine.graph)
        answer = reasoner.answer(text)
        print(format_answer(answer))
        return

    engine = Engine(cfg)

    if cmd == "process":
        text = " ".join(args[1:])
        result = engine.process(text)
        print(f"Fragment: {result['fragment_id']}")
        print(f"Related: {len(result['related'])} fragments")
        for r in result['related'][:3]:
            print(f"  {r['score']:.3f} — {r['metadata']}")
        engine._save_state()

    elif cmd == "query":
        text = " ".join(args[1:])
        result = engine.query(text)
        print(f"Related: {len(result['related'])} fragments")
        for r in result['related'][:5]:
            print(f"  {r['score']:.3f} — {r['metadata']}")

    elif cmd == "refine":
        text = " ".join(args[1:])
        traj = engine.refine(text)
        print(f"Refinement trajectory ({len(traj)} iters):")
        for t in traj:
            print(f"  iter {t['iter']}: active={t['active_nodes']} score={t['top_score']:.3f}")

    elif cmd == "stats":
        g = engine.graph
        print(f"Concepts: {g.n_concepts}")
        print(f"Relations: {g.n_relations}")
        print(f"Avg confidence: {g.avg_confidence:.3f}")
        if g.n_concepts > 0:
            print("\nTop concepts:")
            for name, conf in g.top_concepts(5):
                print(f"  {name}: {conf:.3f}")

    elif cmd == "reset":
        engine.reset()
        print("Engine state reset.")

    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python -m worldfield [reason|process|query|stats|reset]")


if __name__ == "__main__":
    main()
