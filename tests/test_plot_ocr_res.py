import unittest

import numpy as np
import pandas as pd

from img2text import plot_ocr_res


class TestPlotOCRRes(unittest.TestCase):
    def test_plot_ocr_res(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        ocr_res = pd.DataFrame(
            {
                "x0": [10, 50],
                "y0": [20, 60],
                "w": [30, 20],
                "h": [40, 20],
                "text": ["text1", "text2"],
            }
        )
        plot_ocr_res(image, ocr_res)
        # Visual check needed for the plot


if __name__ == "__main__":
    unittest.main()
