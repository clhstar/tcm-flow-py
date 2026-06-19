from pathlib import Path


SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _strip_sql_comment(line: str) -> str:
    return line.split("--", 1)[0].rstrip()


def split_sql_statements(sql: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    for raw_line in sql.splitlines():
        sql_line = _strip_sql_comment(raw_line)
        line = sql_line.strip()
        if not line or line.startswith("--"):
            continue
        current.append(sql_line)
        if line.endswith(";"):
            statement = "\n".join(current).strip()
            statements.append(statement[:-1].strip())
            current = []
    if current:
        statements.append("\n".join(current).strip())
    return statements


async def run_schema_migrations(connection) -> None:
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    async with connection.transaction():
        for statement in split_sql_statements(sql):
            await connection.execute(statement)
