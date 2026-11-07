import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from download import TRIE_SHA, ROOT, cache_tokenizer


class TokenizerCacheTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "models/plamo/tokenizer.jsonl").is_file(), "Download vocabulary first")
    def test_prepare_and_repair(self):
        source = (ROOT / "models/plamo/tokenizer.jsonl").read_bytes()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            path = root / "tokenizer.jsonl"
            cache = root / "tokenizer.jsonl.trie"
            path.write_bytes(source)
            cache_tokenizer(root)
            self.assertEqual(hashlib.sha256(cache.read_bytes()).hexdigest(), TRIE_SHA)
            self.assertEqual(path.read_bytes(), source)
            timestamp = cache.stat().st_mtime_ns
            cache_tokenizer(root)
            self.assertEqual(cache.stat().st_mtime_ns, timestamp)
            cache.write_bytes(b"corrupt")
            cache_tokenizer(root)
            self.assertEqual(hashlib.sha256(cache.read_bytes()).hexdigest(), TRIE_SHA)
            self.assertEqual(path.read_bytes(), source)
            self.assertFalse(cache.with_name(cache.name + ".part").exists())
            cache.write_bytes(b"preserve")
            with patch.object(Path, "replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    cache_tokenizer(root)
            self.assertEqual(cache.read_bytes(), b"preserve")
            self.assertEqual(path.read_bytes(), source)
            self.assertFalse(cache.with_name(cache.name + ".part").exists())

    def test_bad_source_keeps_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "tokenizer.jsonl"
            cache = root / "tokenizer.jsonl.trie"
            source.write_bytes(b"bad source")
            cache.write_bytes(b"preserve")
            with self.assertRaisesRegex(ValueError, "Unexpected tokenizer source"):
                cache_tokenizer(root)
            self.assertEqual(cache.read_bytes(), b"preserve")
            self.assertEqual(source.read_bytes(), b"bad source")
            self.assertFalse(cache.with_name(cache.name + ".part").exists())


if __name__ == "__main__":
    unittest.main()
