"""Contract tests for positional splits, labels, provenance, and input leakage."""

import csv
import importlib.util
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from src.data.derm7pt import Derm7ptDataset
from src.data.labels import ATTRIBUTE_CLASSES, DIAGNOSIS_CLASSES, encode_attribute, encode_diagnosis
from src.data.manifest import audit_manifest, build_manifest, image_relative_path, normalize_image_path_case, prepare_dataset, read_manifest, resolve_image_path, validate_manifest_against_source, write_manifest
from src.data.splits import make_dev_folds, validate_official_indexes
from src.data.transforms import build_transforms

HAS_PIL = importlib.util.find_spec("PIL") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


def make_source(root, repeated_images=False):
    (root / "meta").mkdir()
    (root / "images").mkdir()
    diagnoses = ("clark nevus", "basal cell carcinoma", "melanoma (in situ)")
    rows = []
    for index, diagnosis in enumerate(diagnoses):
        source = {"case_num": str(100 + index), "case_id": "", "diagnosis": diagnosis, "clinic": f"clinic_{index}.png", "derm": f"derm_{index}.png", "sex": "female" if index == 0 else "male", "location": f"location_{index}"}
        source.update({name: "absent" for name in ATTRIBUTE_CLASSES})
        rows.append(source)
        for modality, name in ((0, source["clinic"]), (1, source["derm"])):
            path = root / "images" / name
            if HAS_PIL:
                from PIL import Image
                color = (25, 30, 40) if repeated_images else (25 + index * 20, 30 + modality * 40, 40)
                Image.new("RGB", (4, 4), color).save(path)
            else:
                path.write_bytes(b"fixture-" + str(index).encode() + str(modality).encode())
    with (root / "meta" / "meta.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Unsorted source positions intentionally test positional association.
    for partition, position in (("train", 2), ("valid", 0), ("test", 1)):
        (root / "meta" / f"{partition}_indexes.csv").write_text(f"indexes\n{position}\n", encoding="utf-8")


class LabelContractTests(unittest.TestCase):
    def test_author_diagnosis_order_and_unknown_rejection(self):
        self.assertEqual(DIAGNOSIS_CLASSES, ("BCC", "NEV", "MEL", "MISC", "SK"))
        self.assertEqual(encode_diagnosis("melanoma (less than 0.76 mm)"), 2)
        self.assertEqual(encode_diagnosis("reed or spitz nevus"), 1)
        self.assertEqual(encode_diagnosis("vascular lesion"), 3)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            encode_diagnosis("squamous cell carcinoma")

    def test_missing_attribute_is_not_absent_and_groupings_match_author(self):
        self.assertEqual(encode_attribute("pigment_network", None), (-1, False))
        self.assertEqual(encode_attribute("pigment_network", "absent"), (0, True))
        self.assertEqual(encode_attribute("vascular_structures", "linear irregular"), (2, True))
        self.assertEqual(encode_attribute("regression_structures", "combinations"), (1, True))
        self.assertEqual(sum(map(len, ATTRIBUTE_CLASSES.values())), 19)
        with self.assertRaises(ValueError):
            encode_attribute("pigmentation", "unlisted state")

    def test_partition_duplicates_gaps_bounds_and_default_counts_fail(self):
        for indexes in (
            {"train": [0], "valid": [0], "test": [2]},
            {"train": [0], "valid": [], "test": [2]},
            {"train": [0], "valid": [1], "test": [3]},
        ):
            with self.assertRaises(ValueError):
                validate_official_indexes(indexes, 3, False)
        with self.assertRaisesRegex(ValueError, "counts changed"):
            validate_official_indexes({"train": [0], "valid": [1], "test": [2]}, 3)

    def test_relative_paths_do_not_escape_root(self):
        self.assertEqual(image_relative_path("NEL\\image.JPG"), "images/NEL/image.JPG")
        self.assertEqual(image_relative_path("images/NEL/image.JPG"), "images/NEL/image.JPG")
        for path in ("../image.jpg", "/image.jpg", "C:\\image.jpg"):
            with self.assertRaises(ValueError):
                image_relative_path(path)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        make_source(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_original_positions_and_no_patient_or_substitution_inference(self):
        rows, provenance = build_manifest(self.root, False)
        self.assertEqual([row["official_split"] for row in rows], ["valid", "test", "train"])
        self.assertEqual([row["source_row_index"] for row in rows], [0, 1, 2])
        self.assertEqual(rows[2]["diagnosis_index"], 2)
        self.assertTrue(all(row["patient_id"] is None for row in rows))
        self.assertTrue(all(row["is_clinical_substituted"] == "unknown" for row in rows))
        self.assertEqual(set(provenance["index_sha256"]), {"train", "valid", "test"})

    def test_roundtrip_preserves_types_and_missing_masks(self):
        rows, _ = build_manifest(self.root, False)
        rows[0]["attr_pigment_network"] = -1
        rows[0]["attr_mask_pigment_network"] = False
        path = self.root / "samples.csv"
        write_manifest(rows, path)
        loaded = read_manifest(path)
        self.assertIs(loaded[0]["attr_mask_pigment_network"], False)
        self.assertEqual(loaded[0]["attr_pigment_network"], -1)
        self.assertIsNone(loaded[0]["patient_id"])

    def test_original_source_verification_catches_split_and_attribute_changes(self):
        rows, _ = build_manifest(self.root, False)
        validate_manifest_against_source(rows, self.root, False)
        rows[0]["official_split"], rows[1]["official_split"] = rows[1]["official_split"], rows[0]["official_split"]
        with self.assertRaisesRegex(ValueError, "official_split"):
            validate_manifest_against_source(rows, self.root, False)

    def test_fractional_json_indexes_are_rejected_instead_of_truncated(self):
        rows, _ = build_manifest(self.root, False)
        rows[0]["source_row_index"] = 0.9
        path = self.root / "invalid.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid integer"):
            read_manifest(path)
        rows, _ = build_manifest(self.root, False)
        rows[0]["attr_pigment_network"] = 2
        with self.assertRaisesRegex(ValueError, "attr_pigment_network"):
            validate_manifest_against_source(rows, self.root, False)

    def test_partial_audit_is_not_passed_and_existing_hash_changes_fail(self):
        rows, _ = build_manifest(self.root, False)
        first = audit_manifest(rows, self.root, False, False)
        self.assertFalse(first["passed"])
        self.assertFalse(first["formal_ready"])
        self.assertEqual(first["root"], str(self.root.resolve()))
        self.assertEqual(first["clinical_substitution"]["unknown"], 3)
        resolve_image_path(self.root, rows[0]["clinical_path"]).write_bytes(b"changed")
        second = audit_manifest(rows, self.root, False, False)
        self.assertTrue(any("hash changed" in message for message in second["errors"]))

    def test_selected_missing_modalities_fail_at_dataset_construction(self):
        rows, _ = build_manifest(self.root, False)
        rows[2]["has_clinical"] = False
        with self.assertRaisesRegex(ValueError, "Selected modality missing"):
            Derm7ptDataset(rows, self.root, "train", ("clinical",))
        dataset = Derm7ptDataset(rows, self.root, "train", ("dermoscopic",))
        self.assertEqual(len(dataset), 1)

    def test_path_spelling_is_checked_without_automatic_repair(self):
        rows, _ = build_manifest(self.root, False)
        original = resolve_image_path(self.root, rows[0]["clinical_path"])
        original.rename(original.with_name(original.name.upper()))
        report = audit_manifest(rows, self.root, False, False)
        self.assertTrue(any(issue["issue"] == "path_case_mismatch" for issue in report["image_issues"]))
        self.assertEqual(rows[0]["clinical_path"], "images/clinic_0.png")

    def test_explicit_path_case_normalization_preserves_and_reverifies_source(self):
        original = self.root / "images" / "clinic_0.png"
        original.rename(original.with_name(original.name.upper()))
        rows, _ = build_manifest(self.root, False, normalize_path_case=True)
        self.assertEqual(rows[0]["source_clinical_path"], "clinic_0.png")
        self.assertEqual(rows[0]["clinical_path"], "images/CLINIC_0.PNG")
        validate_manifest_against_source(rows, self.root, False)
        report = audit_manifest(rows, self.root, False, False)
        self.assertEqual(len(report["path_normalization_events"]), 1)
        self.assertEqual(len(report["path_normalization_events"][0]["sha256"]), 64)
        self.assertFalse(any(issue["issue"] == "path_case_mismatch" for issue in report["image_issues"]))
        rows[0]["source_clinical_path"] = "derm_0.png"
        with self.assertRaisesRegex(ValueError, "source_clinical_path"):
            validate_manifest_against_source(rows, self.root, False)

    def test_case_normalization_rejects_ambiguous_matching_files(self):
        (self.root / "images" / "CLINIC_0.PNG").write_bytes(b"other")
        with self.assertRaisesRegex(ValueError, "unique match"):
            normalize_image_path_case(self.root, "clinic_0.png")

    @unittest.skipUnless(HAS_PIL, "Pillow is required for decode audit")
    def test_duplicate_hash_across_partitions_blocks_audit(self):
        rows, _ = build_manifest(self.root, False)
        a = resolve_image_path(self.root, rows[0]["clinical_path"])
        b = resolve_image_path(self.root, rows[1]["dermoscopic_path"])
        b.write_bytes(a.read_bytes())
        report = audit_manifest(rows, self.root, True, False)
        self.assertFalse(report["passed"])
        self.assertTrue(report["cross_partition_same_hash"])
        self.assertEqual(report["clinical_substitution"]["unknown"], 3)

    @unittest.skipUnless(HAS_PIL, "Pillow is required for decode audit")
    def test_prepare_retains_raw_data_and_saves_provenance(self):
        raw = (self.root / "meta" / "meta.csv").read_bytes()
        output = self.root / "manifests"
        report = prepare_dataset(self.root, output, True, False)
        self.assertTrue(report["passed"])
        self.assertEqual(report["status"], "needs_review")
        self.assertFalse(report["formal_ready"])
        self.assertEqual(raw, (self.root / "meta" / "meta.csv").read_bytes())
        self.assertEqual(len(report["manifest_sha256"]), 64)
        self.assertTrue((output / "label_map.json").is_file())
        vocab = json.loads((output / "metadata_vocab.json").read_text())
        self.assertEqual(vocab["fit_partition"], "train")
        self.assertFalse(vocab["input_enabled"])
        self.assertEqual(set(vocab["fields"]), {"sex", "location"})
        self.assertIn("location_2", vocab["fields"]["location"]["vocabulary"])
        self.assertNotIn("location_0", vocab["fields"]["location"]["vocabulary"])
        self.assertNotIn("location_1", vocab["fields"]["location"]["vocabulary"])

    @unittest.skipUnless(HAS_PIL and HAS_TORCH, "PyTorch and Pillow are required for Dataset tensors")
    def test_dataset_batch_contract_excludes_targets_metadata_and_ids_from_inputs(self):
        import torch
        rows, _ = build_manifest(self.root, False)
        dataset = Derm7ptDataset(rows, self.root, "valid", ("clinical",))
        inputs, targets, ids = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
        self.assertEqual(set(inputs), {"clinical", "modality_mask"})
        self.assertEqual(inputs["clinical"].shape, (1, 3, 224, 224))
        self.assertEqual(inputs["modality_mask"].tolist(), [[True, False]])
        self.assertEqual(targets["diagnosis"].dtype, torch.long)
        self.assertEqual(set(targets["attributes"]), set(ATTRIBUTE_CLASSES))
        self.assertEqual(set(ids), {"sample_id", "lesion_id", "source_row_index"})


class TransformAndFoldTests(unittest.TestCase):
    def test_transform_configuration_is_explicit(self):
        self.assertEqual(build_transforms().geometry_coupling, "shared")
        self.assertEqual(build_transforms({"geometry_coupling": "independent"}).geometry_coupling, "independent")
        for config in ({"image_size": 0}, {"geometry_coupling": "guessed"}, {"normalization": "none"}, {"train_augmentation": "random_crop"}, {"appearance_coupling": "shared"}):
            with self.assertRaises(ValueError):
                build_transforms(config)

    def test_dev_folds_leave_official_test_sealed(self):
        rows = [{"sample_id": f"s{index}", "lesion_id": f"l{index}", "source_row_index": index, "official_split": "train" if index < 12 else "test", "diagnosis_index": index % 2} for index in range(14)]
        folds = make_dev_folds(rows, 3, 101)
        self.assertEqual({row["source_row_index"] for row in folds}, set(range(12)))
        self.assertEqual(folds, make_dev_folds(rows, 3, 101))
        self.assertEqual({row["dev_fold"] for row in folds}, {0, 1, 2})

    @unittest.skipUnless(HAS_PIL and HAS_TORCH, "PyTorch and Pillow are required for paired geometry")
    def test_shared_and_independent_geometry_follow_configuration(self):
        import torch
        from PIL import Image
        image = Image.new("RGB", (2, 2))
        image.putdata([(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)])
        images = {"clinical": image, "dermoscopic": image}
        with patch("src.data.transforms.random.random", side_effect=[0.2, 0.9]):
            shared = build_transforms({"image_size": 2}, True)(images)
        self.assertTrue(torch.equal(shared["clinical"], shared["dermoscopic"]))
        with patch("src.data.transforms.random.random", side_effect=[0.2, 0.9, 0.2, 0.9, 0.9, 0.9]):
            independent = build_transforms({"image_size": 2, "geometry_coupling": "independent"}, True)(images)
        self.assertFalse(torch.equal(independent["clinical"], independent["dermoscopic"]))
        plain = build_transforms({"image_size": 2, "train_augmentation": "none"}, True)(images)
        self.assertTrue(torch.equal(plain["clinical"], independent["dermoscopic"]))


if __name__ == "__main__":
    unittest.main()
