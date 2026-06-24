"""WorldField CLI entry point: python -m worldfield [command]

Commands:
  (no args)   Launch interactive CLI
  process     Process a single text input
  query       Query without storing
  benchmark   Run benchmarks
"""
from __future__ import annotations

import sys

from .config import Config
from .core.engine import Engine


def main():
    args = sys.argv[1:]

    if not args:
        from .cli.app import run_cli
        run_cli()
        return

    cmd = args[0]
    cfg = Config()
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

    elif cmd == "benchmark":
        from .benchmarks.runner import run_benchmarks
        run_benchmarks(engine, cfg)

    elif cmd == "reset":
        engine.reset()
        print("Engine state reset.")

    elif cmd == "stats":
        print(f"Fragments: {engine.store.count}")
        print(f"Active slots: {engine.slots.active_count()}/{cfg.n_slots}")
        print(f"Graph edges: {engine.graph.n_edges}")
        print(f"Graph events: {engine.graph.N}")

    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python -m worldfield [process|query|refine|benchmark|stats|reset]")


if __name__ == "__main__":
    main()
