import unittest

from img2text2.engine import get_box, put_line_num


class TestPutLineNum(unittest.TestCase):
    def test_put_line_num(self):
        boxes = [
            [10, 20, 30, 40, "text1"],
            [50, 60, 70, 80, "text2"],
            [10, 100, 30, 120, "text3"],
        ]
        bboxes = get_box(boxes)
        put_line_num(bboxes)
        self.assertIn("line_num", bboxes.columns)
        self.assertGreaterEqual(bboxes["line_num"].min(), 1)


if __name__ == "__main__":
    unittest.main()
