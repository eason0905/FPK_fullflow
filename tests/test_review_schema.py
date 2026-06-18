from __future__ import annotations

import unittest

from real_image_process.FPK_PJ_fullflow.review.schema import slugify


class ReviewSchemaTest(unittest.TestCase):
    def test_slugify_preserves_plus_in_part_number(self) -> None:
        self.assertEqual(slugify("HFCN-7150+"), "HFCN-7150+")
        self.assertNotEqual(slugify("HFCN-7150+"), slugify("HFCN-7150"))

    def test_slugify_preserves_fullwidth_slash_in_part_number(self) -> None:
        self.assertEqual(slugify("ADC128D818CIMTX／NOPB"), "ADC128D818CIMTX／NOPB")
        self.assertNotEqual(slugify("ADC128D818CIMTX／NOPB"), slugify("ADC128D818CIMTX_NOPB"))
        self.assertEqual(slugify("A/B"), "A_B")


if __name__ == "__main__":
    unittest.main()
