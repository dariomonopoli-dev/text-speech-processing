import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_SOURCE = "ci" + "ty"
EXCLUDED_IDENTIFIERS = re.compile(
    rf"swiss-german-{EXCLUDED_SOURCE}-sentences|"
    rf"translation_{EXCLUDED_SOURCE}|{EXCLUDED_SOURCE}50|"
    rf"\b{EXCLUDED_SOURCE} data\b",
    flags=re.IGNORECASE,
)


class SwissCrawlOnlyTests(unittest.TestCase):
    def test_repository_has_no_excluded_dataset_artifacts_or_references(self):
        forbidden_paths = [
            ROOT / "data" / f"translation_{EXCLUDED_SOURCE}50.csv",
            ROOT / "data" / f"translation_prompt_{EXCLUDED_SOURCE}50.txt",
        ]
        self.assertEqual([str(path) for path in forbidden_paths if path.exists()], [])

        matches = []
        for directory in (ROOT / "code", ROOT / "docs"):
            for path in directory.rglob("*"):
                if path.is_file() and path.suffix in {".py", ".md", ".ipynb"}:
                    if EXCLUDED_IDENTIFIERS.search(path.read_text(encoding="utf-8")):
                        matches.append(str(path.relative_to(ROOT)))
        self.assertEqual(matches, [])


if __name__ == "__main__":
    unittest.main()
