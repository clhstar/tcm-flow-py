import asyncio
import json
import unittest
from datetime import timezone
from uuid import UUID

from app.store.postgres_run_manager import PostgresRunManager, _run_from_row
from app.store.postgres_thread_store import PostgresThreadStore, _thread_from_row
from app.store.run_manager import RunManager


class NoGetRow:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        return self.values[key]


class FakeConnection:
    def __init__(self):
        self.fetchrow_calls = []
        self.fetch_calls = []
        self.execute_calls = []
        self.rows = {}

    async def fetchrow(self, sql, *args):
        self.fetchrow_calls.append((sql, args))
        if "insert into app_threads" in sql.lower():
            return {
                "thread_id": args[0],
                "created_at": args[1],
                "updated_at": args[1],
                "status": "idle",
                "metadata": {},
            }
        if "insert into app_runs" in sql.lower():
            return {
                "run_id": args[0],
                "thread_id": args[1],
                "assistant_id": args[2],
                "status": "pending",
                "created_at": args[3],
                "updated_at": args[3],
                "error": None,
            }
        return self.rows.get(args[0])

    async def fetch(self, sql, *args):
        self.fetch_calls.append((sql, args))
        return list(self.rows.values())

    async def execute(self, sql, *args):
        self.execute_calls.append((sql, args))
        return "UPDATE 1"


class FakeAcquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakePool:
    def __init__(self):
        self.connection = FakeConnection()

    def acquire(self):
        return FakeAcquire(self.connection)


class PostgresStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_thread_store_create_returns_thread_record(self):
        pool = FakePool()
        store = PostgresThreadStore(pool)

        record = await store.create()

        UUID(record.thread_id)
        self.assertEqual(record.status, "idle")
        sql, args = pool.connection.fetchrow_calls[0]
        sql = sql.lower()
        self.assertIn("insert into app_threads", sql)
        self.assertIsInstance(args[0], UUID)
        self.assertIs(args[1].tzinfo, timezone.utc)

    async def test_thread_store_update_values_writes_metadata_json(self):
        pool = FakePool()
        store = PostgresThreadStore(pool)

        await store.update_values(
            "00000000-0000-0000-0000-000000000001",
            {"conversation": []},
        )

        sql, args = pool.connection.execute_calls[0]
        self.assertIn("metadata = metadata ||", sql.lower())
        self.assertIsInstance(args[0], UUID)
        self.assertEqual(json.loads(args[1]), {"conversation": []})
        self.assertIs(args[2].tzinfo, timezone.utc)

    async def test_thread_store_update_values_syncs_conversation_to_messages_table(self):
        pool = FakePool()
        store = PostgresThreadStore(pool)

        await store.update_values(
            "00000000-0000-0000-0000-000000000001",
            {
                "conversation": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好，我在。"},
                ]
            },
            run_id="00000000-0000-0000-0000-000000000002",
        )

        sql_texts = [sql.lower() for sql, _ in pool.connection.execute_calls]
        self.assertTrue(any("delete from app_messages" in sql for sql in sql_texts))
        inserts = [
            (sql, args)
            for sql, args in pool.connection.execute_calls
            if "insert into app_messages" in sql.lower()
        ]
        self.assertEqual(len(inserts), 2)
        _, first_args = inserts[0]
        self.assertIsInstance(first_args[0], UUID)
        self.assertIsInstance(first_args[1], UUID)
        self.assertIsNone(first_args[2])
        self.assertEqual(first_args[3], 0)
        self.assertEqual(first_args[4], "human")
        self.assertEqual(first_args[5], "user")
        self.assertEqual(json.loads(first_args[8]), "你好")
        self.assertIsNone(first_args[9])
        self.assertTrue(first_args[10])

    async def test_thread_store_update_values_does_not_sync_raw_agent_messages_table(self):
        pool = FakePool()
        store = PostgresThreadStore(pool)

        await store.update_values(
            "00000000-0000-0000-0000-000000000001",
            {
                "messages": [
                    {
                        "id": "m1",
                        "type": "human",
                        "content": "你好",
                    }
                ]
            },
            run_id="00000000-0000-0000-0000-000000000002",
        )

        sql_texts = [sql.lower() for sql, _ in pool.connection.execute_calls]
        self.assertFalse(any("delete from app_messages" in sql for sql in sql_texts))
        self.assertFalse(any("insert into app_messages" in sql for sql in sql_texts))

    async def test_run_manager_create_returns_run_record(self):
        pool = FakePool()
        manager = PostgresRunManager(pool)

        record = await manager.create(
            "00000000-0000-0000-0000-000000000001",
            "lead_agent",
        )

        UUID(record.run_id)
        self.assertEqual(record.status, "pending")
        self.assertEqual(record.assistant_id, "lead_agent")
        sql, args = pool.connection.fetchrow_calls[0]
        self.assertIn("insert into app_runs", sql.lower())
        self.assertIsInstance(args[0], UUID)
        self.assertIsInstance(args[1], UUID)
        self.assertIs(args[3].tzinfo, timezone.utc)

    async def test_run_manager_set_status_writes_error(self):
        pool = FakePool()
        manager = PostgresRunManager(pool)

        await manager.set_status(
            "00000000-0000-0000-0000-000000000002",
            "error",
            error="boom",
        )

        sql, args = pool.connection.execute_calls[0]
        self.assertIn("update app_runs", sql.lower())
        self.assertIsInstance(args[0], UUID)
        self.assertEqual(args[1], "error")
        self.assertEqual(args[2], "boom")
        self.assertIs(args[3].tzinfo, timezone.utc)

    async def test_in_memory_run_manager_shutdown_waits_for_active_task(self):
        manager = RunManager()
        record = await manager.create(
            "00000000-0000-0000-0000-000000000001",
            "lead_agent",
        )
        completed = []

        async def finish():
            await asyncio.sleep(0)
            completed.append(True)

        record.task = asyncio.create_task(finish())

        await manager.shutdown(timeout=1.0)

        self.assertTrue(record.task.done())
        self.assertEqual(completed, [True])

    async def test_postgres_run_manager_shutdown_waits_for_active_task(self):
        pool = FakePool()
        manager = PostgresRunManager(pool)
        record = await manager.create(
            "00000000-0000-0000-0000-000000000001",
            "lead_agent",
        )
        completed = []

        async def finish():
            await asyncio.sleep(0)
            completed.append(True)

        record.task = asyncio.create_task(finish())

        await manager.shutdown(timeout=1.0)

        self.assertTrue(record.task.done())
        self.assertEqual(completed, [True])

    def test_thread_row_conversion_does_not_require_get(self):
        row = NoGetRow(
            {
                "thread_id": "00000000-0000-0000-0000-000000000001",
                "created_at": "2026-06-18T00:00:00",
                "updated_at": "2026-06-18T00:00:01",
                "status": "idle",
                "metadata": {"conversation": []},
            }
        )

        record = _thread_from_row(row)

        self.assertEqual(record.thread_id, "00000000-0000-0000-0000-000000000001")
        self.assertEqual(record.values, {"conversation": []})

    def test_thread_row_conversion_decodes_json_metadata_text(self):
        row = NoGetRow(
            {
                "thread_id": "00000000-0000-0000-0000-000000000001",
                "created_at": "2026-06-18T00:00:00",
                "updated_at": "2026-06-18T00:00:01",
                "status": "idle",
                "metadata": '{"conversation": []}',
            }
        )

        record = _thread_from_row(row)

        self.assertEqual(record.values, {"conversation": []})

    def test_run_row_conversion_does_not_require_get(self):
        row = NoGetRow(
            {
                "run_id": "00000000-0000-0000-0000-000000000002",
                "thread_id": "00000000-0000-0000-0000-000000000001",
                "assistant_id": "lead_agent",
                "status": "error",
                "created_at": "2026-06-18T00:00:00",
                "updated_at": "2026-06-18T00:00:01",
                "error": "boom",
            }
        )

        record = _run_from_row(row)

        self.assertEqual(record.run_id, "00000000-0000-0000-0000-000000000002")
        self.assertEqual(record.error, "boom")


if __name__ == "__main__":
    unittest.main()
