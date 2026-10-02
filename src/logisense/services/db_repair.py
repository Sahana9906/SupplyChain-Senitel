"""Diagnoses and repairs the specific class of bug where a database's
`alembic_version` stamp says migrations have run, but the tables those
migrations were supposed to create don't actually exist -- e.g. from a
`stamp head` run without a matching `upgrade`, or a migration that
failed partway through in a way that still advanced the stamp.

This is NOT a workaround that bypasses Alembic: the repair works
entirely through Alembic's own stamp + upgrade commands, re-pointed at
the revision that actually matches the database's real state, so the
subsequent `upgrade head` genuinely executes whatever was missed --
through the real migration files, not manual DDL.

How it determines "what tables each revision actually creates": each
migration file's source is parsed (regex on `op.create_table("name"`)
rather than executed, so this works for any past or future migration
without needing a maintained revision->tables mapping.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import inspect
from sqlalchemy.engine import Engine

_CREATE_TABLE_PATTERN = re.compile(r'op\.create_table\(\s*["\'](\w+)["\']')


@dataclass(frozen=True)
class SchemaDiagnosis:
    stamped_revision: str | None
    actual_tables: set[str]
    expected_tables: set[str]
    missing_tables: set[str]
    is_consistent: bool
    repair_target_revision: str | None
    """The revision to re-stamp to before re-running upgrade, if a repair
    is needed and one can be determined. None if already consistent, or
    if no clean repair point could be identified (unusual state --
    surfaced rather than guessed at)."""


@dataclass(frozen=True)
class SchemaRepairResult:
    diagnosis_before: SchemaDiagnosis
    action_taken: str
    diagnosis_after: SchemaDiagnosis


def _tables_created_by_revision(script_dir: ScriptDirectory, revision: str) -> set[str]:
    script = script_dir.get_revision(revision)
    source = open(script.path, encoding="utf-8").read()
    return set(_CREATE_TABLE_PATTERN.findall(source))


def diagnose_schema(engine: Engine, alembic_cfg: Config, expected_tables: set[str]) -> SchemaDiagnosis:
    inspector = inspect(engine)
    actual_tables = set(inspector.get_table_names())

    stamped_revision = None
    if "alembic_version" in actual_tables:
        with engine.connect() as conn:
            row = conn.exec_driver_sql("SELECT version_num FROM alembic_version").first()
            stamped_revision = row[0] if row else None

    missing = expected_tables - actual_tables
    is_consistent = not missing

    repair_target = None
    if missing and stamped_revision:
        script_dir = ScriptDirectory.from_config(alembic_cfg)
        # Walk the migration chain from base to head, tracking the
        # cumulative set of tables each revision *would* have created.
        # The repair target is the last revision whose full cumulative
        # table set is entirely present in the database -- i.e. the
        # last point the database's real state genuinely matches.
        revisions_in_order = list(reversed(list(script_dir.walk_revisions())))
        cumulative: set[str] = set()
        # Start at "base" (nothing applied yet) -- always a valid,
        # honest fallback if not even the first migration's tables exist.
        last_consistent_revision: str | None = "base"
        for rev in revisions_in_order:
            cumulative |= _tables_created_by_revision(script_dir, rev.revision)
            if cumulative.issubset(actual_tables):
                last_consistent_revision = rev.revision
            else:
                break
        repair_target = last_consistent_revision

    return SchemaDiagnosis(
        stamped_revision=stamped_revision, actual_tables=actual_tables, expected_tables=expected_tables,
        missing_tables=missing, is_consistent=is_consistent, repair_target_revision=repair_target,
    )


def repair_schema(engine: Engine, alembic_cfg: Config, expected_tables: set[str]) -> SchemaRepairResult:
    from alembic import command

    before = diagnose_schema(engine, alembic_cfg, expected_tables)

    if before.is_consistent:
        return SchemaRepairResult(diagnosis_before=before, action_taken="No repair needed -- schema already consistent.", diagnosis_after=before)

    if before.repair_target_revision is None:
        raise RuntimeError(
            "Could not determine a safe repair point: the database's tables don't cleanly "
            "match the cumulative table set of any migration in the chain. This needs manual "
            "review rather than an automated repair -- inspect `alembic history` and the "
            "database's actual tables directly."
        )

    # Re-point the stamp at the revision that matches reality, then let
    # a normal `upgrade head` do the rest -- through the real migration
    # files, not manual DDL.
    command.stamp(alembic_cfg, before.repair_target_revision)
    command.upgrade(alembic_cfg, "head")

    after = diagnose_schema(engine, alembic_cfg, expected_tables)
    action = (
        f"Re-stamped from '{before.stamped_revision}' (did not match actual tables) to "
        f"'{before.repair_target_revision}' (last revision whose tables are genuinely present), "
        f"then ran a real 'upgrade head' to create the {len(before.missing_tables)} missing table(s): "
        f"{sorted(before.missing_tables)}."
    )
    return SchemaRepairResult(diagnosis_before=before, action_taken=action, diagnosis_after=after)
