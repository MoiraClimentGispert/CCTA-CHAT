import copy
from pathlib import Path
import unittest
from ccta_chat.data import read_json, prepare
from ccta_chat.evaluation import evaluate, token_scores, legacy_texts

ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.pred = read_json(ROOT/"examples/predictions.json")
        self.ref = read_json(ROOT/"examples/references.json")

    def test_fixture_and_reordering(self):
        result = evaluate(self.pred, self.ref)
        self.assertAlmostEqual(result["token_precision"], .875)
        self.assertAlmostEqual(result["token_recall"], .75)
        self.assertAlmostEqual(result["token_f1"], 19/24)
        pred = copy.deepcopy(self.pred[::-1])
        for record in pred:
            record["turns"].reverse()
        self.assertEqual(evaluate(pred, self.ref), result)

    def test_set_semantics(self):
        self.assertEqual(token_scores("AMBER amber", "amber"), (1, 1, 1))
        self.assertEqual(token_scores("amber,", "amber"), (0, 0, 0))

    def test_missing_and_duplicate_ids(self):
        for changed, message in [(self.pred[:-1], "ID mismatch"),
                                  (self.pred+[self.pred[0]], "Duplicate record_id")]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                evaluate(changed, self.ref)
        self.pred[0]["turns"].append(copy.deepcopy(self.pred[0]["turns"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate turn_id"):
            evaluate(self.pred, self.ref)

    def test_conflicting_prompt_or_scan(self):
        for key in ("question", "scan_id"):
            changed = copy.deepcopy(self.pred)
            if key == "question":
                changed[0]["turns"][0][key] = "A different task?"
            else:
                changed[0][key] = "another-phantom"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Conflicting"):
                evaluate(changed, self.ref)

    def test_missing_ids_are_not_inferred(self):
        del self.pred[0]["record_id"]
        with self.assertRaisesRegex(ValueError, "record_id"):
            evaluate(self.pred, self.ref)

    def test_legacy_schema_is_explicit(self):
        pred = [{"image": None, "conversations_out": [{"answer": "amber"}]}]
        ref = [{"image": None, "conversations": [{"from": "gpt", "value": "amber"}]}]
        p, r = legacy_texts(pred, ref)
        self.assertEqual(p, [str(pred[0])])
        self.assertEqual(r, [str(ref[0])])
        with self.assertRaises(ValueError):
            evaluate(pred, ref)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.rows = read_json(ROOT/"examples/references.json")

    def test_markers_and_multiturn_identity(self):
        self.rows[1]["image"] = "phantom-b.npz"
        prepared = prepare(self.rows)
        image = prepared[1]
        self.assertEqual(image["type"], "long_answer")
        self.assertEqual(sum(m["value"].count("<image>") for m in image["conversations"]), 1)
        self.assertIn("<provided>", image["conversations"][0]["value"])
        self.assertEqual([m["turn_id"] for m in image["conversations"]], ["0", "0", "1", "1"])
        self.assertNotIn("<image>", prepared[0]["conversations"][0]["value"])

    def test_literature_stays_text_only(self):
        row = self.rows[0]
        row.update(source="literature", article_id="independent-example", image="unrelated.npz")
        with self.assertRaisesRegex(ValueError, "text-only"):
            prepare([row])
        row["image"] = None
        self.assertEqual(prepare([row])[0]["article_id"], "independent-example")
        del row["article_id"]
        with self.assertRaisesRegex(ValueError, "article_id"):
            prepare([row])

    def test_same_source_cannot_cross_splits(self):
        first, second = self.rows[:2]
        for row in (first, second):
            row.update(source="literature", article_id="same-article", image=None)
        first["split"], second["split"] = "train", "validation"
        with self.assertRaisesRegex(ValueError, "different splits"):
            prepare([first, second])


if __name__ == "__main__":
    unittest.main()
