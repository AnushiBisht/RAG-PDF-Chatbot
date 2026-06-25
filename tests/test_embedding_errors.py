import unittest

from src.embeddings import format_embedding_error


class EmbeddingErrorFormattingTests(unittest.TestCase):
    def test_includes_troubleshooting_hints(self):
        message = format_embedding_error(RuntimeError("network timeout"))
        self.assertIn("embedding model", message.lower())
        self.assertIn("huggingface", message.lower())
        self.assertIn("offline", message.lower())


if __name__ == "__main__":
    unittest.main()
