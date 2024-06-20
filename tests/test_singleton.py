import unittest

from invoice2text.img2text.utils.singleton import SingletonByParams


class ExampleClass(metaclass=SingletonByParams):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class SingletonTestCase(unittest.TestCase):
    def test_singleton_pattern(self):
        instance1 = ExampleClass(
            use_gpu=True, psm=6, paddle_lang="french", tesseract_lang="eng+fra+ara"
        )
        instance2 = ExampleClass(
            use_gpu=False, psm=7, paddle_lang="english", tesseract_lang="eng"
        )

        # Verify instances are different for different configurations
        self.assertIsNot(instance1, instance2)

        # Verify instances are reused for the same configuration
        instance3 = ExampleClass(
            use_gpu=True, psm=6, paddle_lang="french", tesseract_lang="eng+fra+ara"
        )
        self.assertIs(instance1, instance3)


if __name__ == "__main__":
    unittest.main()
