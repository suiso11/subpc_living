"""Focused tests for scripts/backup.sh / scripts/restore.sh (I1 PostgreSQL).

Static assertions on the shell sources plus isolated end-to-end runs that use
a FAKE `docker` executable on PATH. These tests never touch a real database,
container, network, or real .env files.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKUP_SH = REPO_ROOT / "scripts" / "backup.sh"
RESTORE_SH = REPO_ROOT / "scripts" / "restore.sh"
COMPOSE_FILE = REPO_ROOT / "compose.yaml"

BASH = None


def _find_bash():
    """Find a usable POSIX bash; avoid WSL's System32 bash.exe on Windows."""
    candidates = []
    for cand in (
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        "/usr/bin/bash",
        "/bin/bash",
    ):
        if Path(cand).is_file():
            return cand
    exe = shutil.which("bash")
    lowered = (exe or "").lower()
    if exe and "system32" not in lowered and "windowsapps" not in lowered:
        return exe
    return None


BASH = _find_bash()
FAKE_DUMP = b"PGDMP-FAKE-CUSTOM-FORMAT-DUMP-BYTES"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _bash_path(value) -> str:
    """Convert an absolute Windows path for Git Bash/MSYS commands."""
    text = str(value)
    if os.name == "nt" and len(text) >= 3 and text[1] == ":" and text[2] in ("\\", "/"):
        return f"/{text[0].lower()}/{text[3:].replace(chr(92), '/')}"
    return text


class BackupRestoreStaticTests(unittest.TestCase):
    """Source-level assertions (no execution)."""

    def setUp(self):
        self.backup_src = _read(BACKUP_SH)
        self.restore_src = _read(RESTORE_SH)

    # -- backup.sh ---------------------------------------------------------

    def test_backup_supports_off_auto_required_mode(self):
        self.assertIn("POSTGRES_BACKUP_MODE", self.backup_src)
        for mode in ("off", "auto", "required"):
            self.assertIn(mode, self.backup_src)
        # invalid values must be rejected
        self.assertIn("must be off, auto, or required", self.backup_src)

    def test_backup_uses_custom_format_pg_dump_via_compose_postgres(self):
        self.assertIn("pg_dump", self.backup_src)
        self.assertIn("--format=custom", self.backup_src)
        self.assertRegex(self.backup_src, r"docker compose .*exec -T postgres")

    def test_backup_does_not_handle_credentials(self):
        src = self.backup_src
        self.assertNotIn("POSTGRES_PASSWORD", src)
        self.assertNotIn("PGPASSWORD", src)
        # dump command must rely on the container's own environment
        self.assertIn('"$POSTGRES_USER"', src)
        self.assertIn('"$POSTGRES_DB"', src)

    def test_backup_dump_enters_manifest_flow(self):
        # same .archive_list pipe format consumed by the manifest generator
        self.assertIn(
            "postgres|postgres.dump|${bytes}|1", self.backup_src
        )

    def test_backup_is_backward_compatible_by_default(self):
        self.assertNotRegex(
            self.backup_src,
            r'POSTGRES_BACKUP_MODE="\$\{[^}]*:-required\}"',
        )
        self.assertIn('POSTGRES_BACKUP_MODE="${POSTGRES_BACKUP_MODE:-auto}"', self.backup_src)
        self.assertIn("compose postgres service not running", self.backup_src)

    def test_backup_sanitizes_manifest_string_fields(self):
        """manifest.json string fields must strip CR/LF/control chars so a
        Windows `sha256sum --version` (CRLF) cannot corrupt the JSON."""
        src = self.backup_src
        self.assertIn("tr -d '\\000-\\037'", src)
        for field in (
            "BASH_VERSION_STR",
            "TAR_VERSION",
            "SHA256_VERSION",
            "HOSTNAME",
        ):
            self.assertIn(field + '="$(sanitize_for_json', src)

    # -- restore.sh --------------------------------------------------------

    def test_restore_requires_explicit_opt_in_flag(self):
        self.assertIn("--restore-postgres", self.restore_src)
        self.assertIn("RESTORE_POSTGRES=0", self.restore_src)  # default off

    def test_restore_uses_required_pg_restore_flags(self):
        self.assertIn("pg_restore", self.restore_src)
        self.assertRegex(
            self.restore_src,
            r"pg_restore\s+--clean\s+--if-exists\s+--no-owner\s+--no-privileges",
        )
        self.assertRegex(self.restore_src, r"docker compose .*exec -T postgres")

    def test_verify_only_precedes_and_blocks_pg_restore(self):
        src = self.restore_src
        verify_exit = src.index("Verify-only mode: skipping extraction.")
        pg_restore_call = src.index("exec pg_restore")
        self.assertLess(verify_exit, pg_restore_call)
        # verify-only exits 0 before any mutation is possible
        self.assertRegex(src, r"Verify-only mode: skipping extraction\.\",?\s*\n\s*exit 0")

    def test_manifest_parser_fails_closed(self):
        self.assertIn("python3 is required to verify manifest.json", self.restore_src)
        self.assertIn("integrity verification aborted", self.restore_src)
        self.assertNotIn("using limited parsing", self.restore_src)

    def test_restore_fails_safely_without_prerequisites(self):
        src = self.restore_src
        # missing dump check happens before pg_restore call
        missing_check = src.index("not found in ${BACKUP_DIR}; nothing to restore")
        pg_restore_call = src.index("exec pg_restore")
        self.assertLess(missing_check, pg_restore_call)
        self.assertIn("compose postgres service is not running", src)

    def test_neither_script_contains_credentials(self):
        for src in (self.backup_src, self.restore_src):
            self.assertNotIn("POSTGRES_PASSWORD", src)
            self.assertNotIn("PGPASSWORD", src)


@unittest.skipUnless(BASH, "bash not available on this machine")
class FakeDockerEndToEndTests(unittest.TestCase):
    """Isolated end-to-end runs with a fake `docker` executable on PATH."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="i1-pg-backup-test-"))
        self.bin_dir = self.tmp / "bin"
        self.bin_dir.mkdir()
        self.log_file = self.tmp / "fake_docker.log"
        self.stdin_capture = self.tmp / "fake_docker_stdin.out"
        self._write_fake_docker()
        self._maybe_write_python3_shim()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- helpers -----------------------------------------------------------

    def _write_fake_docker(self):
        script = self.bin_dir / "docker"
        script.write_text(
            "#!/usr/bin/env bash\n"
            'printf \'%s\\n\' "$*" >> "$FAKE_DOCKER_LOG"\n'
            'case "$*" in\n'
            '  *"ps --services"*) echo "postgres" ;;\n'
            '  *"pg_dump"*) printf \'%s\' "$FAKE_DUMP_CONTENT" ;;\n'
            '  *"pg_restore"*) cat > "$FAKE_STDIN_CAPTURE" ;;\n'
            "esac\n"
            "exit 0\n",
            encoding="utf-8",
        )
        script.chmod(script.stat().st_mode | 0o111)

    def _maybe_write_python3_shim(self):
        """restore.sh parses the manifest with python3; provide one if absent."""
        if os.name != "nt" and shutil.which("python3"):
            return
        real = Path(sys.executable).as_posix()
        shim = self.bin_dir / "python3"
        shim.write_text(
            f'#!/usr/bin/env bash\nexec "{real}" "$@"\n', encoding="utf-8"
        )
        shim.chmod(shim.stat().st_mode | 0o111)

    def _script_env(self):
        env = os.environ.copy()
        env["PATH"] = str(self.bin_dir) + os.pathsep + env.get("PATH", "")
        env["FAKE_DOCKER_LOG"] = _bash_path(self.log_file)
        env["FAKE_STDIN_CAPTURE"] = _bash_path(self.stdin_capture)
        env["FAKE_DUMP_CONTENT"] = FAKE_DUMP.decode("ascii")
        return env

    def _run_bash(self, script: Path, *args: str, extra_env=None):
        env = self._script_env()
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            [BASH, _bash_path(script), *(_bash_path(arg) for arg in args)],
            cwd=str(self.tmp),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def _make_repo_skeleton(self):
        root = self.tmp / "repo"
        (root / "scripts").mkdir(parents=True)
        (root / "backups").mkdir(parents=True)
        shutil.copy2(BACKUP_SH, root / "scripts" / "backup.sh")
        shutil.copy2(RESTORE_SH, root / "scripts" / "restore.sh")
        shutil.copy2(COMPOSE_FILE, root / "compose.yaml")
        return root

    def _fake_log(self):
        if not self.log_file.exists():
            return ""
        return self.log_file.read_text(encoding="utf-8")

    # -- backup.sh ---------------------------------------------------------

    def test_backup_auto_creates_dump_and_manifest_entry(self):
        root = self._make_repo_skeleton()
        target = self.tmp / "out"
        result = self._run_bash(
            root / "scripts" / "backup.sh",
            "--target-dir", str(target),
            "--keep-daily", "1",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        ts_dirs = [d for d in target.iterdir() if d.is_dir()]
        self.assertEqual(len(ts_dirs), 1)
        backup_dir = ts_dirs[0]
        dump = backup_dir / "postgres.dump"
        self.assertTrue(dump.is_file())
        self.assertEqual(dump.read_bytes(), FAKE_DUMP)

        manifest = json.loads((backup_dir / "manifest.json").read_text(encoding="utf-8"))
        entries = {a["name"]: a for a in manifest["archives"]}
        self.assertIn("postgres", entries)
        entry = entries["postgres"]
        self.assertEqual(entry["file"], "postgres.dump")
        self.assertEqual(entry["bytes"], len(FAKE_DUMP))
        self.assertEqual(
            entry["sha256"], hashlib.sha256(dump.read_bytes()).hexdigest()
        )
        # no credentials anywhere in the manifest or fake docker invocations
        self.assertNotIn("password", json.dumps(manifest).lower())
        self.assertNotIn("PASSWORD", self._fake_log())

    def test_backup_required_fails_when_postgres_unavailable(self):
        root = self._make_repo_skeleton()
        # a docker fake whose ps reports nothing => service "not running"
        docker = self.bin_dir / "docker"
        docker.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        target = self.tmp / "out-required"
        result = self._run_bash(
            root / "scripts" / "backup.sh",
            "--target-dir", str(target),
            "--keep-daily", "1",
            extra_env={"POSTGRES_BACKUP_MODE": "required"},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(target.glob("*/postgres.dump")) if target.exists() else False)

    def test_backup_off_never_invokes_docker(self):
        root = self._make_repo_skeleton()
        target = self.tmp / "out-off"
        result = self._run_bash(
            root / "scripts" / "backup.sh",
            "--target-dir", str(target),
            "--keep-daily", "1",
            extra_env={"POSTGRES_BACKUP_MODE": "off"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("pg_dump", self._fake_log())

    def test_backup_rejects_invalid_mode(self):
        root = self._make_repo_skeleton()
        result = self._run_bash(
            root / "scripts" / "backup.sh",
            "--target-dir", str(self.tmp / "out-bad"),
            extra_env={"POSTGRES_BACKUP_MODE": "sometimes"},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be off, auto, or required", result.stderr)

    def test_backup_dry_run_does_not_create_dump(self):
        root = self._make_repo_skeleton()
        result = self._run_bash(
            root / "scripts" / "backup.sh",
            "--dry-run",
            "--target-dir", str(self.tmp / "out-dry"),
            extra_env={"POSTGRES_BACKUP_MODE": "required"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[DRY]", result.stdout)
        self.assertNotIn("pg_dump", self._fake_log())

    # -- manifest hardening (CR/LF) ----------------------------------------

    def _write_crlf_bash_env(self):
        """Write a BASH_ENV file that injects CRLF-emitting shell functions for
        sha256sum/tar/hostname, mirroring Windows GNU coreutils behavior that
        previously corrupted manifest.json.

        Unlike PATH-based fake executables, BASH_ENV function injection is
        deterministic on Git Bash/MSYS: shell functions take precedence over
        PATH lookups, and only ``--version``/hostname emit CRLF. Normal
        sha256sum/tar processing delegates to the real binary via ``command``,
        so the manifest's sha256 digest stays correct.
        """
        bash_env = self.tmp / "crlf_bash_env.sh"
        bash_env.write_text(
            "# BASH_ENV: CRLF regression overrides (no PATH fakes).\n"
            "_crlf_sha256sum() {\n"
            '    if [[ "${1:-}" == "--version" ]]; then\n'
            "        printf 'sha256sum (GNU coreutils) 9.5\\r\\n'\n"
            "        return 0\n"
            "    fi\n"
            '    command sha256sum "$@"\n'
            "}\n"
            "_crlf_tar() {\n"
            '    if [[ "${1:-}" == "--version" ]]; then\n'
            "        printf 'tar (GNU tar) 1.35\\r\\n'\n"
            "        return 0\n"
            "    fi\n"
            '    command tar "$@"\n'
            "}\n"
            "_crlf_hostname() {\n"
            "    printf 'windows-host\\r\\n'\n"
            "}\n"
            'sha256sum() { _crlf_sha256sum "$@"; }\n'
            'tar() { _crlf_tar "$@"; }\n'
            'hostname() { _crlf_hostname "$@"; }\n'
            "export -f sha256sum tar hostname "
            "_crlf_sha256sum _crlf_tar _crlf_hostname\n",
            encoding="utf-8",
        )
        return bash_env

    def test_backup_manifest_valid_when_tool_versions_emit_crlf(self):
        """Regression: Windows sha256sum/tar --version and hostname can emit
        CRLF. manifest.json must remain strict-parseable JSON with no raw CR.

        Uses BASH_ENV function injection (not PATH-based fakes) so the test is
        deterministic on Git Bash/MSYS: functions override the real commands,
        emit CRLF only for --version/hostname, and delegate actual sha256sum/tar
        work to the real binaries via ``command`` — the manifest's sha256 digest
        therefore matches the true hash of postgres.dump.
        """
        root = self._make_repo_skeleton()
        bash_env = self._write_crlf_bash_env()
        target = self.tmp / "out-crlf"
        result = self._run_bash(
            root / "scripts" / "backup.sh",
            "--target-dir", str(target),
            "--keep-daily", "1",
            extra_env={"BASH_ENV": _bash_path(bash_env)},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        ts_dirs = [d for d in target.iterdir() if d.is_dir()]
        self.assertEqual(len(ts_dirs), 1)
        backup_dir = ts_dirs[0]
        manifest_path = backup_dir / "manifest.json"
        raw = manifest_path.read_bytes()
        self.assertNotIn(b"\r", raw)
        manifest = json.loads(raw.decode("utf-8"))
        self.assertEqual(
            manifest["generated_by"]["sha256sum"],
            "sha256sum (GNU coreutils) 9.5",
        )
        self.assertEqual(manifest["generated_by"]["tar"], "tar (GNU tar) 1.35")
        self.assertEqual(manifest["hostname"], "windows-host")
        entries = {a["name"]: a for a in manifest["archives"]}
        self.assertIn("postgres", entries)
        dump = backup_dir / "postgres.dump"
        self.assertEqual(
            entries["postgres"]["sha256"],
            hashlib.sha256(dump.read_bytes()).hexdigest(),
        )
        self.assertNotIn("password", json.dumps(manifest).lower())

    # -- restore.sh ----------------------------------------------------------

    def _make_backup_dir(self, name: str, include_dump=True, corrupt_sha=False):
        backup_dir = self.tmp / "backups" / name
        backup_dir.mkdir(parents=True)
        archives = []
        if include_dump:
            data = FAKE_DUMP
            digest = hashlib.sha256(data).hexdigest()
            if corrupt_sha:
                digest = "0" * 64
            (backup_dir / "postgres.dump").write_bytes(data)
            archives.append({
                "name": "postgres",
                "file": "postgres.dump",
                "sha256": digest,
                "bytes": len(data),
            })
        (backup_dir / "manifest.json").write_text(
            json.dumps({"schema_version": 1, "archives": archives}),
            encoding="utf-8",
        )
        return backup_dir

    def test_restore_postgres_streams_dump_with_required_flags(self):
        root = self._make_repo_skeleton()
        backup_dir = self._make_backup_dir("ts-ok")
        restore_target = self.tmp / "restored"
        result = self._run_bash(
            root / "scripts" / "restore.sh",
            str(backup_dir),
            "--target", str(restore_target),
            "--restore-postgres",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.stdin_capture.exists())
        self.assertEqual(self.stdin_capture.read_bytes(), FAKE_DUMP)
        log = self._fake_log()
        self.assertIn("--clean", log)
        self.assertIn("--if-exists", log)
        self.assertIn("--no-owner", log)
        self.assertIn("--no-privileges", log)
        self.assertIn("postgres", log)
        self.assertNotIn("PASSWORD", log)

    def test_restore_verify_only_never_mutates_postgres(self):
        root = self._make_repo_skeleton()
        backup_dir = self._make_backup_dir("ts-verify")
        result = self._run_bash(
            root / "scripts" / "restore.sh",
            str(backup_dir),
            "--target", str(self.tmp / "restored-verify"),
            "--verify-only",
            "--restore-postgres",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("pg_restore", self._fake_log())
        self.assertFalse((self.tmp / "restored-verify").exists())

    def test_restore_missing_dump_fails_before_pg_restore(self):
        root = self._make_repo_skeleton()
        backup_dir = self._make_backup_dir("ts-nodump", include_dump=False)
        result = self._run_bash(
            root / "scripts" / "restore.sh",
            str(backup_dir),
            "--target", str(self.tmp / "restored-nodump"),
            "--restore-postgres",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("nothing to restore", result.stderr)
        self.assertNotIn("pg_restore", self._fake_log())

    def test_restore_corrupt_dump_aborts_before_pg_restore(self):
        root = self._make_repo_skeleton()
        backup_dir = self._make_backup_dir("ts-corrupt", corrupt_sha=True)
        result = self._run_bash(
            root / "scripts" / "restore.sh",
            str(backup_dir),
            "--target", str(self.tmp / "restored-corrupt"),
            "--restore-postgres",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("failed integrity check", result.stdout)
        self.assertNotIn("pg_restore", self._fake_log())

    def test_restore_without_flag_leaves_postgres_alone_but_notifies(self):
        root = self._make_repo_skeleton()
        backup_dir = self._make_backup_dir("ts-noflag")
        result = self._run_bash(
            root / "scripts" / "restore.sh",
            str(backup_dir),
            "--target", str(self.tmp / "restored-noflag"),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("pg_restore", self._fake_log())
        self.assertIn("--restore-postgres", result.stdout)


if __name__ == "__main__":
    unittest.main()
