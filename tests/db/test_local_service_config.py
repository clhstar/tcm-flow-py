import unittest
from pathlib import Path


class LocalServiceConfigTests(unittest.TestCase):
    def test_docker_compose_defines_postgres_and_elasticsearch(self):
        text = Path("docker-compose.persistence.yml").read_text(encoding="utf-8")

        self.assertIn("postgres", text)
        self.assertIn("pgvector/pgvector", text)
        self.assertIn("elasticsearch", text)
        self.assertIn("9200:9200", text)

    def test_env_example_contains_persistence_settings(self):
        text = Path(".env.example").read_text(encoding="utf-8")

        self.assertIn("DATABASE_URL=", text)
        self.assertIn("CHECKPOINT_BACKEND=postgres", text)
        self.assertIn("RAG_ENGINE=database", text)
        self.assertIn("ELASTICSEARCH_URL=", text)


if __name__ == "__main__":
    unittest.main()
