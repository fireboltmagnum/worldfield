import unittest
from pathlib import Path
import csv
import importlib


ROOT = Path(__file__).resolve().parents[1]


class RepoHealthTests(unittest.TestCase):
    def test_root_requirements_covers_all_experiment_families(self):
        req = ROOT / "requirements.txt"
        self.assertTrue(req.exists(), "repo-level requirements.txt should exist")
        text = req.read_text(encoding="utf-8").lower()
        for package in ["torch", "numpy", "matplotlib", "faiss-cpu", "scipy", "scikit-learn"]:
            self.assertIn(package, text)

    def test_shared_project_metadata_package_exists(self):
        init_file = ROOT / "worldfield" / "__init__.py"
        project_file = ROOT / "worldfield" / "project.py"
        self.assertTrue(init_file.exists(), "worldfield package should exist")
        self.assertTrue(project_file.exists(), "worldfield.project should exist")
        text = project_file.read_text(encoding="utf-8")
        self.assertIn("DAY_DIRECTORIES", text)
        self.assertIn("required_packages", text)

    def test_shared_worldfield_helpers_cover_common_experiment_needs(self):
        expected_modules = {
            "worldfield.config": ["ExperimentConfig"],
            "worldfield.devices": ["pick_device"],
            "worldfield.loaders": ["checkpoint_path", "artifact_dir"],
            "worldfield.metrics": ["precision_recall_f1", "retrieval_recall_at_1"],
            "worldfield.graphs": ["edge", "undirected_edges"],
        }
        for module_name, symbols in expected_modules.items():
            module = importlib.import_module(module_name)
            for symbol in symbols:
                self.assertTrue(hasattr(module, symbol), f"{module_name}.{symbol} missing")

    def test_gitignore_filters_python_and_generated_runtime_files(self):
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for pattern in ["__pycache__/", "*.py[cod]", ".pytest_cache/", ".mypy_cache/", "day_*/out/"]:
            self.assertIn(pattern, text)

    def test_generated_artifacts_are_split_into_reports_artifacts(self):
        artifact_root = ROOT / "reports" / "artifacts"
        self.assertTrue((artifact_root / "day_one" / "worldfield.pt").exists())
        self.assertTrue((artifact_root / "day_one" / "worldfield_rich.pt").exists())
        self.assertTrue((artifact_root / "day_one" / "latent_space.png").exists())
        self.assertTrue((artifact_root / "day_three" / "world_state.png").exists())
        self.assertFalse((ROOT / "day_one" / "out").exists(), "day_one/out should not hold generated artifacts")

    def test_full_explanation_covers_teacher_facing_topics(self):
        doc = ROOT / "WORLD_FIELD_FULL_EXPLANATION.md"
        self.assertTrue(doc.exists(), "full explanation document should exist")
        text = doc.read_text(encoding="utf-8").lower()
        for phrase in [
            "direct multimodality",
            "experiment map",
            "long short-term memory spiking",
            "lsnn",
            "day 1",
            "day 9",
            "diagram",
            "specification",
        ]:
            self.assertIn(phrase, text)

    def test_key_saved_verdicts_are_machine_readable_smoke_checks(self):
        edge_csv = ROOT / "day_nine" / "reports" / "graph_recovery" / "edge_recovery.csv"
        self.assertTrue(edge_csv.exists())
        rows = list(csv.DictReader(edge_csv.open(newline="", encoding="utf-8")))
        triple = [
            row for row in rows
            if row["world"] == "triple_confound" and row["learner"] == "pure_cmi_k3"
        ]
        self.assertEqual(len(triple), 1)
        self.assertGreaterEqual(float(triple[0]["f1"]), 0.99)

        findings = (ROOT / "FINDINGS.md").read_text(encoding="utf-8")
        self.assertIn("R@1 ≈ 0.99", findings)
        self.assertIn("precision@10 = 1.000", findings)
        self.assertIn("0.92 purity", findings)

    def test_day9_naming_is_scoped_to_concept_level_skeleton_recovery(self):
        paths = [
            ROOT / "day_nine" / "day9d_causal_graph_recovery.py",
            ROOT / "day_nine" / "reports" / "graph_recovery" / "DAY9D_CAUSAL_GRAPH_REPORT.md",
            ROOT / "WORLD_FIELD_FULL_EXPLANATION.md",
        ]
        for path in paths:
            text = path.read_text(encoding="utf-8").lower()
            self.assertIn("concept-level causal skeleton", text, str(path))
            self.assertIn("not fragment-scale", text, str(path))


if __name__ == "__main__":
    unittest.main()
