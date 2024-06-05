import unittest

from img2text2.engine import get_box


class TestGetBox(unittest.TestCase):
    def test_get_box(self):
        boxes = [[10, 20, 30, 40, "text1"], [50, 60, 70, 80, "text2"]]
        df = get_box(boxes)
        self.assertEqual(df.shape[0], 2)
        self.assertListEqual(
            df.columns.tolist(),
            [
                "left",
                "top",
                "x1",
                "y1",
                "text",
                "height",
                "width",
                "x0",
                "y0",
                "h",
                "w",
                "c0",
                "c1",
            ],
        )
        self.assertEqual(df.iloc[0]["text"], "text1")
        self.assertEqual(df.iloc[1]["text"], "text2")


if __name__ == "__main__":
    unittest.main()
