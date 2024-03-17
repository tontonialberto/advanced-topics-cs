from unittest import TestCase
from unittest.mock import Mock

from app.domain.similarity.cached import CachedSimilarity
from app.domain.similarity.similarity import Similarity

class TestCachedSimilarity(TestCase):
    def test_inherit_name_from_inner_similarity(self) -> None:
        similarity_name = "MySimilarity"
        similarity = Mock(spec=Similarity)
        similarity.name = similarity_name
        
        cached_similarity = CachedSimilarity(similarity=similarity)
        
        self.assertEqual(similarity_name, cached_similarity.name)
