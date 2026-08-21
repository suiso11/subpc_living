#!/usr/bin/env bash
# restore.sh — homelab restore for subpc_living
#
# Usage: restore.sh <backup_timestamp_dir> --target <restore_root> [--verify-only]
#
# Verifies sha256 integrity of every archive against manifest.json BEFORE
# extracting anything. On mismatch, exits non-zero without modifying the target.
#
# Extraction preserves repo-root-relative layout under --target.

set -euo pipefail

# ── Resolve REPO_ROOT from this script's location ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Parse arguments ──
VERIFY_ONLY=0
BACKUP_TS_DIR=""
TARGET=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --verify-only) VERIFY_ONLY=1; shift ;;
        --target)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --target requires a directory argument" >&2
                exit 1
            fi
            TARGET="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 <backup_timestamp_dir> --target <restore_root> [--verify-only]"
            echo ""
            echo "Options:"
            echo "  --target DIR      Restore root directory (required)"
            echo "  --verify-only     Verify sha256 without extracting"
            echo ""
            echo "Example:"
            echo "  $0 20260822-143000 --target /tmp/subpc-restore"
            echo "  $0 20260822-143000 --target /tmp/subpc-restore --verify-only"
            exit 0
            ;;
        -*) echo "Unknown option: $1" >&2; exit 1 ;;
        *)
            if [[ -z "$BACKUP_TS_DIR" ]]; then
                BACKUP_TS_DIR="$1"
            else
                echo "Unexpected argument: $1" >&2; exit 1
            fi
            shift ;;
    esac
done

if [[ -z "$BACKUP_TS_DIR" ]]; then
    echo "Error: backup_timestamp_dir is required" >&2
    exit 1
fi
if [[ -z "$TARGET" ]]; then
    echo "Error: --target <restore_root> is required" >&2
    exit 1
fi

# Resolve backup directory
# Accept absolute path or bare timestamp name (look in default backups dir)
if [[ -d "$BACKUP_TS_DIR" ]]; then
    BACKUP_DIR="$BACKUP_TS_DIR"
else
    BACKUP_DIR="${REPO_ROOT}/backups/${BACKUP_TS_DIR}"
fi

if [[ ! -d "$BACKUP_DIR" ]]; then
    echo "Error: backup directory not found: ${BACKUP_DIR}" >&2
    exit 1
fi

MANIFEST="${BACKUP_DIR}/manifest.json"
if [[ ! -f "$MANIFEST" ]]; then
    echo "Error: manifest.json not found in ${BACKUP_DIR}" >&2
    exit 1
fi

echo "=== subpc_living Restore ==="
echo "  Backup : ${BACKUP_DIR}"
echo "  Target : ${TARGET}"
echo "  Mode   : $([ $VERIFY_ONLY -eq 1 ] && echo 'verify-only' || echo 'verify + extract')"
echo ""

# ── Verify phase ──
echo "--- Verification Phase ---"

PASS_COUNT=0
FAIL_COUNT=0
ARCHIVE_NAMES=()

# Parse manifest.json without jq (using python3 as a safe parser)
# If python3 is not available, fall back to simple grep parsing
read_manifest_archives() {
    python3 -c "
import json, sys
with open('${MANIFEST}') as f:
    data = json.load(f)
for a in data.get('archives', []):
    print(f\"{a['name']}|{a['file']}|{a['sha256']}|{a['bytes']}\")
" 2>/dev/null || {
        # Fallback: grep-based parsing (less robust)
        grep -oP '"name"\s*:\s*"\K[^"]+' "$MANIFEST" | head -20
        echo "WARNING: python3 not available, using limited parsing" >&2
    }
}

while IFS='|' read -r name file sha256 expected_bytes; do
    name="$(echo "$name" | xargs)"
    file="$(echo "$file" | xargs)"
    sha256="$(echo "$sha256" | xargs)"
    expected_bytes="$(echo "$expected_bytes" | xargs)"

    # 古いmanifestのskip/dryエントリ防御 (検証対象外として扱う)
    if [[ "$file" == "skip" || "$file" == "dry" ]]; then
        echo "  [SKIP] ${name}: no archive (group had no data)"
        continue
    fi

    archive_path="${BACKUP_DIR}/${file}"
    ARCHIVE_NAMES+=("$name")

    if [[ ! -f "$archive_path" ]]; then
        echo "  [FAIL] ${name}: archive file not found (${file})"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    # Compute actual sha256
    actual_sha256="$(sha256sum "$archive_path" 2>/dev/null | awk '{print $1}')"

    # Compare
    if [[ "$actual_sha256" == "$sha256" ]]; then
        actual_bytes="$(stat -c%s "$archive_path" 2>/dev/null || stat -f%z "$archive_path" 2>/dev/null || echo 0)"
        echo "  [PASS] ${name}  sha256=${actual_sha256:0:16}...  size=${actual_bytes}bytes"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "  [FAIL] ${name}"
        echo "         expected: ${sha256}"
        echo "         actual  : ${actual_sha256}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done < <(read_manifest_archives)

echo ""
echo "Verification result: ${PASS_COUNT} PASS, ${FAIL_COUNT} FAIL"

if [[ $FAIL_COUNT -gt 0 ]]; then
    echo ""
    echo "ERROR: ${FAIL_COUNT} archive(s) failed integrity check."
    echo "No extraction performed. Check backup corruption or incomplete backup."
    exit 1
fi

# ── Verify-only mode: stop here ──
if [[ $VERIFY_ONLY -eq 1 ]]; then
    echo ""
    echo "Verify-only mode: skipping extraction."
    exit 0
fi

# ── Extract phase ──
echo ""
echo "--- Extraction Phase ---"

# Ensure target directory exists
mkdir -p "${TARGET}"

# Extract each archive
for name in "${ARCHIVE_NAMES[@]}"; do
    archive_path="${BACKUP_DIR}/${name}.tar.gz"
    if [[ ! -f "$archive_path" ]]; then
        echo "  [SKIP] ${name}: file not found (was it skipped in backup?)"
        continue
    fi
    echo "  [EXTRACT] ${name}.tar.gz -> ${TARGET}/"
    tar xzf "$archive_path" -C "${TARGET}"
done

echo ""
echo "--- Extraction Complete ---"

# ── Post-restore checklist ──
echo ""
echo "=== Post-Restore Checklist ==="
echo ""

echo "1. Restart systemd user services:"
# Discover actual unit files from scripts/systemd/
SYSTEMD_DIR="${REPO_ROOT}/scripts/systemd"
if [[ -d "$SYSTEMD_DIR" ]]; then
    for unit_file in "${SYSTEMD_DIR}"/*.service; do
        if [[ -f "$unit_file" ]]; then
            unit_name="$(basename "$unit_file")"
            echo "   systemctl --user restart ${unit_name}"
        fi
    done
    # Also list timers
    for timer_file in "${SYSTEMD_DIR}"/*.timer; do
        if [[ -f "$timer_file" ]]; then
            timer_name="$(basename "$timer_file")"
            echo "   systemctl --user restart ${timer_name}"
        fi
    done
fi
echo ""

echo "2. SQLite integrity checks:"
for db in tasks.db growth.db system_metrics.db; do
    echo "   sqlite3 \"${TARGET}/data/*/\"*.db  'PRAGMA integrity_check;'"
done
# More specific paths:
echo "   sqlite3 '${TARGET}/data/tasks/tasks.db'    'PRAGMA integrity_check;'"
echo "   sqlite3 '${TARGET}/data/growth/growth.db'  'PRAGMA integrity_check;'"
echo "   sqlite3 '${TARGET}/data/metrics/system_metrics.db' 'PRAGMA integrity_check;'"
echo ""

echo "3. IMPORTANT: Real .env files are NOT in backups."
echo "   You must manually restore:"
echo "     config/.env          (or appropriate env file)"
echo "     config/discord.env"
echo "   These contain secrets and are excluded from backup by design."
echo ""

echo "4. Verify Ollama models are available:"
echo "   ollama list"
echo "   (If models are missing, re-pull with: ollama pull <model>)"
echo ""

echo "5. Check service health:"
echo "   systemctl --user status subpc-web.service"
echo "   systemctl --user status subpc-discord.service"
echo "   systemctl --user status subpc-voice.service"
echo ""

echo "6. Set correct ownership on restored files:"
echo "   chown -R \$(id -u):\$(id -g) '${TARGET}/data/'"
echo ""

echo "=== Restore Complete ==="
