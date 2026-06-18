from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from real_image_process.FPK_PJ_fullflow.review.adapters.multiview import build_multiview_review


class MultiviewReviewTests(unittest.TestCase):
    def test_build_multiview_review_writes_risk_and_view_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FPK_PJ_fullflow"
            multiview_root = root / "runs" / "run1" / "outputs" / "multiview"
            part_dir = multiview_root / "parts" / "PART"
            part_dir.mkdir(parents=True)
            (part_dir / "unified_multiview_layers.json").write_text(
                json.dumps(
                    {
                        "part_number": "PART",
                        "source_views": ["bottom", "land"],
                        "canonical_source_views": ["bottom", "land"],
                        "package_pads": [{"source_object_id": 1}],
                        "land_pads": [{"source_object_id": 2}],
                        "lead_contacts": [],
                        "multiview_overlay": {
                            "coordinate_mode": "dimension_scaled_centered",
                            "frame": [-1.0, -1.0, 1.0, 1.0],
                            "layers": [
                                {
                                    "raw_view": "bottom",
                                    "canonical_view": "bottom",
                                    "graph_path": "/tmp/PART/bottom.package_graph.json",
                                    "coordinate_mode": "dimension_scaled_centered",
                                    "source_frame": [0.0, 0.0, 2.0, 2.0],
                                    "normalized_frame": [-1.0, -1.0, 1.0, 1.0],
                                    "unit_scales": {"x": 1.0, "y": 1.0, "source": "accepted_dimensions"},
                                    "objects": [
                                        {
                                            "role": "package_pad",
                                            "label": "rect",
                                            "source_label": "rect",
                                            "bbox": [-1.0, -1.0, 1.0, 1.0],
                                            "coordinate_mode": "dimension_scaled_centered",
                                        }
                                    ],
                                }
                            ],
                            "extra_objects": [],
                        },
                        "dimensions": [],
                        "ignored_evidence": [],
                        "missing_canonical_views": ["lateral", "lead_detail"],
                        "source_selection": {
                            "package_pads": {
                                "primary_view": "bottom",
                                "selected_raw_view": "bottom",
                                "used_fallback": False,
                            }
                        },
                        "pad_matching": {"status": "matched"},
                        "conflicts": [],
                        "evidence_summary": {
                            "dimension_value_source_counts": {"table_lookup": 1},
                            "evidence_type_counts": {"package_graph": 2, "scan_result_format": 1},
                        },
                        "summary": {
                            "risk_score": 8.0,
                            "risk_level": "low",
                            "risk_reasons": ["missing views: lateral, lead_detail"],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (part_dir / "evidence.json").write_text(
                json.dumps(
                    {
                        "part_number": "PART",
                        "summary": {
                            "dimension_value_source_counts": {"table_lookup": 1},
                            "evidence_type_counts": {"package_graph": 2, "scan_result_format": 1},
                        },
                        "evidence_refs": [],
                        "ignored_evidence": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (part_dir / "conflicts.json").write_text("[]\n", encoding="utf-8")
            (part_dir / "unified_multiview_layers.svg").write_text("<svg></svg>\n", encoding="utf-8")

            output_root = root / "runs" / "run1" / "outputs" / "review" / "multiview"
            result = build_multiview_review(
                multiview_root=multiview_root,
                output_root=output_root,
                fullflow_root=root,
                run_id="run1",
            )

            self.assertEqual(result["total_items"], 1)
            self.assertTrue((output_root / "index.html").exists())
            self.assertTrue((output_root / "by_risk" / "low.html").exists())
            self.assertTrue((output_root / "by_view" / "bottom.html").exists())
            self.assertTrue((output_root / "by_view" / "land.html").exists())
            self.assertTrue((output_root / "data" / "notes.json").exists())
            self.assertTrue((output_root / "data" / "cases.json").exists())
            cases = json.loads((output_root / "data" / "cases.json").read_text(encoding="utf-8"))
            self.assertEqual(
                cases[0]["metadata"]["source_selection"]["package_pads"]["selected_raw_view"],
                "bottom",
            )
            self.assertEqual(
                cases[0]["metrics"]["source_selection"]["package_pads"]["primary_view"],
                "bottom",
            )
            self.assertEqual(
                cases[0]["metrics"]["evidence_summary"]["dimension_value_source_counts"],
                {"table_lookup": 1},
            )
            self.assertEqual(
                cases[0]["metadata"]["evidence_summary"]["evidence_type_counts"],
                {"package_graph": 2, "scan_result_format": 1},
            )
            media_labels = [item["label"] for item in cases[0]["media"]]
            self.assertIn("Main-view overlay", media_labels)
            self.assertIn("Multi-view overlay", media_labels)
            self.assertTrue((part_dir / "top_bottom_land_overlay.svg").exists())
            self.assertTrue((part_dir / "multi_view_overlay.svg").exists())


if __name__ == "__main__":
    unittest.main()
