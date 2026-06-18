from __future__ import annotations

import unittest

from real_image_process.FPK_PJ_fullflow.review.builders.yolo_errors import (
    YoloBox,
    infer_reason,
    match_boxes,
)


class YoloErrorReviewTests(unittest.TestCase):
    def test_match_boxes_requires_same_class_and_iou_threshold(self) -> None:
        gt_boxes = [
            YoloBox(cls_id=1, xyxy=(0.0, 0.0, 10.0, 10.0)),
            YoloBox(cls_id=2, xyxy=(20.0, 20.0, 30.0, 30.0)),
        ]
        pred_boxes = [
            YoloBox(cls_id=1, xyxy=(1.0, 1.0, 11.0, 11.0), conf=0.9),
            YoloBox(cls_id=3, xyxy=(20.0, 20.0, 30.0, 30.0), conf=0.9),
        ]

        matched_gt, matched_pred = match_boxes(gt_boxes, pred_boxes, 0.5)

        self.assertEqual(matched_gt, [0])
        self.assertEqual(matched_pred, [0])

    def test_infer_reason_splits_missing_extra_and_mixed(self) -> None:
        self.assertEqual(infer_reason(True, False), "missing_detection")
        self.assertEqual(infer_reason(False, True), "extra_detection")
        self.assertEqual(infer_reason(True, True), "missing_and_extra_detection")


if __name__ == "__main__":
    unittest.main()
