import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from topomem_sglang_rsi.rebind_cases import FORMAL_IDS, FORMAL_KIND, INDICES, ORIGINAL_CASE_SET, file_sha, formal_subset


class GatedSubsetTest(unittest.TestCase):
    def test_formal_subset_binds_the_prespecified_original_positions(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "case_set_manifest.json"
            child = root / "formal_case_set_manifest.json"
            cases = {split: [{"id": name} for name in names] for split, names in FORMAL_IDS.items()}
            cases["train"].append({"id": "unused-train"})
            cases["validation"].append({"id": "unused-validation"})

            def bind():
                parent.write_text(json.dumps({"prior_case_set_manifest_sha256": ORIGINAL_CASE_SET, "cases": cases}))
                expected = {split: [cases[split][i] for i in indices] for split, indices in INDICES.items()}
                child.write_text(json.dumps({"kind": FORMAL_KIND, "preregistered_zero_based_indices": INDICES,
                    "prior_case_set_manifest_sha256": ORIGINAL_CASE_SET, "parent_case_set_manifest_sha256": file_sha(parent),
                    "cases": expected}))
                return expected

            expected = bind()
            self.assertEqual(formal_subset(child)["cases"], expected)
            cases["train"].reverse()
            bind()
            with self.assertRaisesRegex(ValueError, "formal SGLang subset"):
                formal_subset(child)


if __name__ == "__main__":
    unittest.main()
