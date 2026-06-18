from __future__ import annotations

import re
import unittest
from pathlib import Path

from real_image_process.FPK_PJ_fullflow.review.common.gallery_renderer import page_relative_url
from real_image_process.FPK_PJ_fullflow.review.common.gallery_renderer import render_card
from real_image_process.FPK_PJ_fullflow.review.common.schemas import ReviewItem, ReviewMedia


class ReviewGalleryRendererTests(unittest.TestCase):
    def test_render_card_groups_source_views_above_gt_result_media(self) -> None:
        item = ReviewItem(
            case_id="case",
            title="PART",
            rank=1,
            part_number="PART",
            file_name="alignment.json",
            view="gt_alignment",
            risk_score=50.0,
            risk_level="medium",
            risk_reasons=["lead_count_mismatch"],
            media=[
                ReviewMedia(label="Source top (top)", path="/tmp/source.png", url="source.png"),
                ReviewMedia(label="Postprocessed top package graph", path="/tmp/top.png", url="top.png"),
                ReviewMedia(label="GT reference", path="/tmp/gt.svg", url="gt.svg"),
                ReviewMedia(label="Aligned result", path="/tmp/result.svg", url="result.svg"),
                ReviewMedia(label="GT vs result", path="/tmp/comparison.svg", url="comparison.svg"),
            ],
        )

        html = render_card(item)

        self.assertIn("media-section--source", html)
        self.assertIn("media-section--postprocessed", html)
        self.assertIn("media-section--result", html)
        self.assertLess(html.index("Source views"), html.index("Main postprocessed views"))
        self.assertLess(html.index("Main postprocessed views"), html.index("GT / Result / Comparison"))
        self.assertLess(html.index("Source top (top)"), html.index("GT reference"))
        self.assertLess(html.index("Postprocessed top package graph"), html.index("GT reference"))

    def test_page_relative_url_resolves_workspace_asset_from_html_page(self) -> None:
        url = "real_image_process/FPK_PJ_fullflow/review/common/static/review.css"
        page_dir = Path.cwd() / "real_image_process/FPK_PJ_fullflow/runs/test/outputs/review/final_comparison/by_status"

        resolved = page_relative_url(url, page_dir)

        self.assertEqual(resolved, "/review/common/static/review.css")

    def test_page_relative_url_roots_fullflow_paths_from_nested_pages(self) -> None:
        page_dir = Path.cwd() / "real_image_process/FPK_PJ_fullflow/runs/test/outputs/review/final_comparison/by_status"

        self.assertEqual(
            page_relative_url("review/common/static/review.css", page_dir),
            "/review/common/static/review.css",
        )
        self.assertEqual(
            page_relative_url("runs/test/outputs/review/final_comparison/data/notes.json", page_dir),
            "/runs/test/outputs/review/final_comparison/data/notes.json",
        )

    def test_render_card_uses_workspace_root_media_and_link_paths(self) -> None:
        media_url = "real_image_process/FPK_PJ_fullflow/review/common/static/review.css"
        page_dir = Path.cwd() / "real_image_process/FPK_PJ_fullflow/runs/test/outputs/review/final_comparison/by_status"
        item = ReviewItem(
            case_id="case",
            title="PART",
            rank=1,
            part_number="PART",
            file_name="alignment.json",
            view="gt_alignment",
            risk_score=50.0,
            risk_level="medium",
            risk_reasons=["aligned"],
            media=[ReviewMedia(label="GT reference", path="/tmp/gt.svg", url=media_url)],
            links={"asset": media_url},
        )

        html = render_card(item, page_dir=page_dir)
        media_src = re.search(r'<img src="([^"]+)"', html)
        link_href = re.search(r'<a href="([^"]+)" target="_blank">asset</a>', html)

        self.assertIsNotNone(media_src)
        self.assertIsNotNone(link_href)
        self.assertEqual(media_src.group(1), "/review/common/static/review.css")
        self.assertEqual(link_href.group(1), "/review/common/static/review.css")

    def test_render_card_shows_score_diagnostic_details(self) -> None:
        item = ReviewItem(
            case_id="case",
            title="PART",
            rank=1,
            part_number="PART",
            file_name="alignment.json",
            view="gt_alignment",
            risk_score=100.0,
            risk_level="high",
            risk_reasons=["scan_result_geometry_fallback"],
            media=[],
            metrics={
                "score_diagnostic_details": [
                    {
                        "reason": "scan_result_geometry_fallback",
                        "metric": "source_independence_score",
                        "value": 0.0,
                        "threshold": 1.0,
                        "stage_hint": "low_score_scan_result_geometry_fallback",
                        "fallback_role_counts": {"land": 2, "package_pad": 2},
                    }
                ]
            },
        )

        html = render_card(item)

        self.assertIn("Risk details", html)
        self.assertIn("source_independence_score", html)
        self.assertIn("low_score_scan_result_geometry_fallback", html)
        self.assertIn("&quot;package_pad&quot;: 2", html)

    def test_render_card_groups_dimension_scaled_graph_with_gt_reference(self) -> None:
        item = ReviewItem(
            case_id="case",
            title="PART",
            rank=1,
            part_number="PART",
            file_name="alignment.json",
            view="gt_alignment",
            risk_score=0.0,
            risk_level="low",
            risk_reasons=["aligned"],
            media=[
                ReviewMedia(label="Dimension-scaled graph", path="/tmp/rotation.svg", url="rotation.svg"),
                ReviewMedia(label="GT reference", path="/tmp/gt.svg", url="gt.svg"),
            ],
        )

        html = render_card(item)

        self.assertIn("Dimension-scaled graph / GT reference", html)
        self.assertLess(html.index("Dimension-scaled graph"), html.index("GT reference"))


if __name__ == "__main__":
    unittest.main()
