"""Tests for tools/v2_log_archive.py — dry-run safety, apply roundtrip, exclusions."""

import gzip
from datetime import datetime, timedelta, timezone

from tools import v2_log_archive as ar


def day_str(days_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y%m%d")


def make_logs(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    files = {
        "old_dated": logs / f"trades_paper_{day_str(20)}.csv",
        "boundary": logs / f"trades_paper_{day_str(14)}.csv",     # exactly N days: kept
        "recent_dated": logs / f"trades_closed_{day_str(2)}.csv",
        "today_dated": logs / f"trades_closed_{day_str(0)}.csv",
        "master": logs / "trades_closed.csv",                      # undated: never touched
        "signals": logs / "live_signals.csv",
        "outlog": logs / "live_executor.out",
    }
    for i, p in enumerate(files.values()):
        p.write_text(f"content-{i}\n" * 10, encoding="utf-8")
    # A file already inside archive/ must never be re-scanned.
    arch = logs / "archive" / day_str(40)[:6]
    arch.mkdir(parents=True)
    files["already_archived"] = arch / f"trades_paper_{day_str(40)}.csv"
    files["already_archived"].write_text("archived\n", encoding="utf-8")
    return logs, files


def snapshot(logs):
    return {str(p.relative_to(logs)): p.read_bytes()
            for p in sorted(logs.rglob("*")) if p.is_file()}


def test_dry_run_changes_nothing(tmp_path, capsys):
    logs, files = make_logs(tmp_path)
    before = snapshot(logs)
    code = ar.main(["--logs-dir", str(logs), "--days", "14"])
    assert code == 0
    assert snapshot(logs) == before                      # byte-identical filesystem
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert files["old_dated"].name in out                # planned
    assert files["recent_dated"].name not in out         # not planned


def test_apply_archives_old_and_roundtrips(tmp_path):
    logs, files = make_logs(tmp_path)
    original = files["old_dated"].read_bytes()
    code = ar.main(["--logs-dir", str(logs), "--days", "14", "--apply"])
    assert code == 0
    assert not files["old_dated"].exists()
    gz = logs / "archive" / day_str(20)[:6] / (files["old_dated"].name + ".gz")
    assert gz.exists()
    with gzip.open(gz, "rb") as f:
        assert f.read() == original                      # lossless


def test_exclusions_survive_apply(tmp_path):
    logs, files = make_logs(tmp_path)
    ar.main(["--logs-dir", str(logs), "--days", "14", "--apply"])
    for key in ("boundary", "recent_dated", "today_dated", "master",
                "signals", "outlog", "already_archived"):
        assert files[key].exists(), f"{key} must not be archived"


def test_cutoff_is_strict(tmp_path):
    logs, files = make_logs(tmp_path)
    cands = ar.find_candidates(logs, 14)
    names = {p.name for p in cands}
    assert files["old_dated"].name in names
    assert files["boundary"].name not in names           # == cutoff day: kept


def test_verify_failure_keeps_original_and_exits_1(tmp_path, monkeypatch):
    logs, files = make_logs(tmp_path)

    def boom(path, archive_root):
        raise IOError("verification failed (simulated)")

    monkeypatch.setattr(ar, "archive_one", boom)
    code = ar.main(["--logs-dir", str(logs), "--days", "14", "--apply"])
    assert code == 1
    assert files["old_dated"].exists()                   # original retained


def test_empty_logs_dir_ok(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    assert ar.main(["--logs-dir", str(logs), "--days", "14"]) == 0
