import unittest

from src.chat.quality import is_low_value_memory_text, should_store_rag_turn
from src.memory.rag import RAGRetriever


class QualityFilterTest(unittest.TestCase):
    def test_detects_repeated_silence_template(self):
        text = "[neutral]\n………。\n\n……。\n\n……。\n\nまた、これですか。\n\nさっさと、寝なさい。"
        self.assertTrue(is_low_value_memory_text(text))
        self.assertFalse(should_store_rag_turn("少女", text))

    def test_allows_normal_informative_answer(self):
        text = "[neutral]\nComfyUIなら、まずCheckpoint LoaderとKSamplerをつなぎ、解像度を低めにして試しなさい。"
        self.assertFalse(is_low_value_memory_text(text))
        self.assertTrue(should_store_rag_turn("comfyuiの使い方", text))


class _FakeVectorStore:
    def __init__(self):
        self.stored = []

    def is_initialized(self):
        return True

    def search_all(self, query, n_results):
        return [
            {
                "distance": 0.1,
                "document": "ユーザー: ん\nAI: ……。\n\n……。\n\n……。\n\nさっさと、寝なさい。",
                "metadata": {"assistant_message": "……。\n\n……。\n\n……。\n\nさっさと、寝なさい。"},
            },
            {
                "distance": 0.2,
                "document": "ユーザー: comfyui\nAI: ComfyUIのワークフローを確認しなさい。",
                "metadata": {"assistant_message": "ComfyUIのワークフローを確認しなさい。"},
            },
            {
                "distance": 0.3,
                "document": "睡眠メモ: 今日はもう寝なさい、という生活ルール。",
                "metadata": {"type": "knowledge", "category": "habit"},
            },
        ]

    def store_conversation_turn(self, **kwargs):
        self.stored.append(kwargs)
        return "doc-1"

    def get_stats(self):
        return {}


class RAGQualityFilterTest(unittest.TestCase):
    def test_retrieve_filters_low_value_memories(self):
        rag = RAGRetriever(_FakeVectorStore())
        results = rag.retrieve("少女")
        self.assertEqual(len(results), 2)
        self.assertIn("ComfyUI", results[0]["document"])
        self.assertEqual(results[1]["metadata"]["type"], "knowledge")

    def test_store_turn_skips_low_value_assistant_message(self):
        store = _FakeVectorStore()
        rag = RAGRetriever(store)
        doc_id = rag.store_turn("ん", "……。\n\n……。\n\n……。\n\nさっさと、寝なさい。")
        self.assertIsNone(doc_id)
        self.assertEqual(store.stored, [])


if __name__ == "__main__":
    unittest.main()
