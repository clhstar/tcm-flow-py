#!/usr/bin/env bash
set -eu

consultation_db="${TCM_CONSULTATION_POSTGRES_DB:-tcm_consultation}"
consultation_user="${TCM_CONSULTATION_POSTGRES_USER:-tcm_app}"
consultation_password="${TCM_CONSULTATION_POSTGRES_PASSWORD:-tcm_app_dev_password}"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
  -v consultation_db="$consultation_db" \
  -v consultation_user="$consultation_user" \
  -v consultation_password="$consultation_password" <<'SQL'
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'consultation_user', :'consultation_password')
WHERE NOT EXISTS (
  SELECT 1 FROM pg_roles WHERE rolname = :'consultation_user'
)\gexec

SELECT format('ALTER ROLE %I LOGIN PASSWORD %L', :'consultation_user', :'consultation_password')\gexec

SELECT format('CREATE DATABASE %I OWNER %I', :'consultation_db', :'consultation_user')
WHERE NOT EXISTS (
  SELECT 1 FROM pg_database WHERE datname = :'consultation_db'
)\gexec
SQL

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$consultation_db" \
  -v consultation_user="$consultation_user" <<'SQL'
GRANT USAGE, CREATE ON SCHEMA public TO :"consultation_user";
SQL
