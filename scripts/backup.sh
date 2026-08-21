#!/usr/bin/env bash
# backup.sh — homelab backup for subpc_living
#
# Usage: backup.sh [--dry-run] [--target-dir DIR] [--keep-daily N]
#
# Creates timestamped archive groups under TARGET_DIR/<timestamp>/ so partial
# restores are possible. Each group is a tar.gz covering one logical data area.
#
# Assumptions (verify via grep if paths change):
#   - Chat history dir: data/chat_history  (src/chat/session.py default)
#   - Vector store dir: data/vectordb      (src/memory/vectorstore.py default)
#   - Profile dir:      data/profile/
#   - Diary dir:        data/diary/
#   - Metrics DB:       data/metrics/system_metrics.db
#   - Growth DB:        data/growth/growth.db
#   - Tasks DB:         data/tasks/tasks.db
#   - Calendar data:    data/calendar/
#
# Override via environment:
#   BACKUP_EXTRA_INCLUDES="path1 path2"   (space-separated, repo-root-relative)
#   BACKUP_EXTRA_EXCLUDES="pattern1 pattern2"
#
# Excluded everywhere:
#   *.psd, tmp audio, camera frames, screen captures,
#   __pycache__, *.pyc, logs older than rotation,
#   model weights (*.gguf, *.onnx, models/)
#
# NEVER backs up: real .env files, config/*.env, config/discord.env

set -euo pipefail

# ── Resolve REPO_ROOT from this script's location ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Defaults ──
TARGET_DIR="${REPO_ROOT}/backups"
KEEP_DAILY=7
DRY_RUN=0

# ── Parse arguments ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1; shift ;;
        --target-dir) TARGET_DIR="$2"; shift 2 ;;
        --keep-daily) KEEP_DAILY="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--target-dir DIR] [--keep-daily N]"
            echo ""
            echo "Options:"
            echo "  --dry-run          Print planned actions without writing"
            echo "  --target-dir DIR   Base directory for backups (default: <repo>/backups)"
            echo "  --keep-daily N     Retention: keep newest N timestamp dirs (default: 7)"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP_DIR="${TARGET_DIR}/${TIMESTAMP}"
MANIFEST="${BACKUP_DIR}/manifest.json"

# ── Global exclusion patterns ──
COMMON_EXCLUDES=(
    --exclude='*.psd'
    --exclude='__pycache__'
    --exclude='*.pyc'
    --exclude='*.pyo'
    --exclude='models/'
    --exclude='*.gguf'
    --exclude='*.onnx'
    --exclude='*.safetensors'
    --exclude='*.bin'
    --exclude='node_modules/'
    --exclude='.venv/'
    --exclude='*.log'
    --exclude='tmp_audio'
    --exclude='tmp_audio/**'
    --exclude='camera_frames'
    --exclude='camera_frames/**'
    --exclude='screen_captures'
    --exclude='screen_captures/**'
    --exclude='*.env'
    --exclude='config/discord.env'
)

# Extra user-defined exclusions
if [[ -n "${BACKUP_EXTRA_EXCLUDES:-}" ]]; then
    for pat in ${BACKUP_EXTRA_EXCLUDES}; do
        COMMON_EXCLUDES+=(--exclude="$pat")
    done
fi

# ── Tool versions for manifest ──
BASH_VERSION_STR="${BASH_VERSION}"
TAR_VERSION="$(tar --version 2>/dev/null | head -1 || echo 'unknown')"
SHA256_VERSION="$(sha256sum --version 2>/dev/null | head -1 || echo 'unknown')"
HOSTNAME="$(hostname)"

# ── Helper: archive a directory group ──
# Args: archive_name dir1 [dir2 ...]
archive_group() {
    local name="$1"; shift
    local archive_path="${BACKUP_DIR}/${name}.tar.gz"
    local exists_dirs=()

    for d in "$@"; do
        if [[ -d "${REPO_ROOT}/${d}" ]]; then
            exists_dirs+=("$d")
        fi
    done

    if [[ ${#exists_dirs[@]} -eq 0 ]]; then
        echo "  [SKIP] ${name}: none of the target directories exist"
        echo "    ${name}|skip|0|0" >> "${BACKUP_DIR}/.archive_list"
        return
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  [DRY]  ${name}.tar.gz:"
        for d in "${exists_dirs[@]}"; do
            echo "    du -sh ${REPO_ROOT}/${d}"
            du -sh "${REPO_ROOT}/${d}" 2>/dev/null || echo "    (unable to measure)"
        done
        echo "    ${name}|dry|0|0" >> "${BACKUP_DIR}/.archive_list"
        return
    fi

    echo "  [ARCHIVE] ${name}.tar.gz"
    tar czf "$archive_path" \
        "${COMMON_EXCLUDES[@]}" \
        -C "$REPO_ROOT" \
        "${exists_dirs[@]}"

    local size bytes file_count
    size="$(stat -c%s "$archive_path" 2>/dev/null || stat -f%z "$archive_path" 2>/dev/null || echo 0)"
    bytes="$size"
    # Count files in the tar (excluding directories)
    file_count="$(tar tzf "$archive_path" 2>/dev/null | grep -cv '/$' || echo 0)"

    echo "    ${name}|${name}.tar.gz|${bytes}|${file_count}" >> "${BACKUP_DIR}/.archive_list"
}

# ── Helper: archive files matching a pattern ──
archive_files() {
    local name="$1"
    local archive_path="${BACKUP_DIR}/${name}.tar.gz"
    shift

    local files=()
    for f in "$@"; do
        if [[ -e "${REPO_ROOT}/${f}" ]]; then
            files+=("$f")
        fi
    done

    if [[ ${#files[@]} -eq 0 ]]; then
        echo "  [SKIP] ${name}: no matching files found"
        echo "    ${name}|skip|0|0" >> "${BACKUP_DIR}/.archive_list"
        return
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  [DRY]  ${name}.tar.gz:"
        for f in "${files[@]}"; do
            echo "    ${REPO_ROOT}/${f}"
        done
        echo "    ${name}|dry|0|0" >> "${BACKUP_DIR}/.archive_list"
        return
    fi

    echo "  [ARCHIVE] ${name}.tar.gz"
    tar czf "$archive_path" \
        "${COMMON_EXCLUDES[@]}" \
        -C "$REPO_ROOT" \
        "${files[@]}"

    local size bytes file_count
    size="$(stat -c%s "$archive_path" 2>/dev/null || stat -f%z "$archive_path" 2>/dev/null || echo 0)"
    bytes="$size"
    file_count="$(tar tzf "$archive_path" 2>/dev/null | grep -cv '/$' || echo 0)"

    echo "    ${name}|${name}.tar.gz|${bytes}|${file_count}" >> "${BACKUP_DIR}/.archive_list"
}

# ── Main ──
echo "=== subpc_living Backup ==="
echo "  Timestamp : ${TIMESTAMP}"
echo "  Target    : ${TARGET_DIR}"
echo "  Keep daily: ${KEEP_DAILY}"
echo "  Dry run   : ${DRY_RUN}"
echo "  Repo root : ${REPO_ROOT}"
echo ""

if [[ $DRY_RUN -eq 0 ]]; then
    mkdir -p "${BACKUP_DIR}"
    echo -n "" > "${BACKUP_DIR}/.archive_list"
fi

echo "Archive groups:"
archive_group "tasks"                    "data/tasks"
archive_group "conversations"           "data/chat_history"
archive_group "rag"                     "data/vectordb"
archive_group "profile_diary"           "data/profile" "data/diary"
archive_group "metrics_calendar_growth" "data/metrics" "data/growth" "data/calendar"
archive_group "config_systemd"          "config" "scripts/systemd"

# Extra includes
if [[ -n "${BACKUP_EXTRA_INCLUDES:-}" ]]; then
    archive_files "extra_includes" ${BACKUP_EXTRA_INCLUDES}
fi

echo ""

# ── Build manifest ──
if [[ $DRY_RUN -eq 1 ]]; then
    echo "=== Dry Run Complete ==="
    echo "  No archives written. Review the planned actions above."
    if [[ -d "${BACKUP_DIR}" ]]; then
        rmdir "${BACKUP_DIR}" 2>/dev/null || true
    fi
    exit 0
fi

echo "Generating manifest.json..."

# Build JSON arrays
ARCHIVE_ENTRIES=""
ARCHIVE_COUNT=0
while IFS='|' read -r name file bytes count; do
    name="$(echo "$name" | xargs)"
    file="$(echo "$file" | xargs)"
    bytes="$(echo "$bytes" | xargs)"
    count="$(echo "$count" | xargs)"

    # .archive_list の2列目(file欄)が skip/dry。name ではなくこちらを見る。
    if [[ "$file" == "skip" || "$file" == "dry" ]]; then
        continue
    fi

    # Compute sha256
    sha256="$(sha256sum "${BACKUP_DIR}/${file}" 2>/dev/null | awk '{print $1}')"

    if [[ $ARCHIVE_COUNT -gt 0 ]]; then
        ARCHIVE_ENTRIES+=","
    fi
    ARCHIVE_ENTRIES+=$(cat <<ENTRY
    {
      "name": "${name}",
      "file": "${file}",
      "sha256": "${sha256}",
      "bytes": ${bytes},
      "file_count": ${count}
    }
ENTRY
)
    ARCHIVE_COUNT=$((ARCHIVE_COUNT + 1))
done < "${BACKUP_DIR}/.archive_list"

# Write manifest.json
cat > "${MANIFEST}" <<MANIFEST_EOF
{
  "schema_version": 1,
  "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hostname": "${HOSTNAME}",
  "archives": [
${ARCHIVE_ENTRIES}
  ],
  "generated_by": {
    "bash": "${BASH_VERSION_STR}",
    "tar": "${TAR_VERSION}",
    "sha256sum": "${SHA256_VERSION}"
  }
}
MANIFEST_EOF

rm -f "${BACKUP_DIR}/.archive_list"

# ── Retention pruning ──
echo ""
echo "Retention pruning (keeping newest ${KEEP_DAILY})..."
BACKUP_BASE="${TARGET_DIR}"
if [[ -d "${BACKUP_BASE}" ]]; then
    # List timestamp dirs sorted newest-first
    mapfile -t TS_DIRS < <(
        find "${BACKUP_BASE}" -maxdepth 1 -mindepth 1 -type d -name '20*' | \
        sort -r
    )

    REMOVE_COUNT=0
    if [[ ${#TS_DIRS[@]} -gt ${KEEP_DAILY} ]]; then
        for (( i=KEEP_DAILY; i<${#TS_DIRS[@]}; i++ )); do
            old_dir="${TS_DIRS[$i]}"
            echo "  Removing old backup: $(basename "$old_dir")"
            rm -rf "$old_dir"
            REMOVE_COUNT=$((REMOVE_COUNT + 1))
        done
    fi
    echo "  Removed ${REMOVE_COUNT} old backup(s)"
fi

# ── Summary ──
echo ""
echo "=== Backup Summary ==="
echo "  Archive dir : ${BACKUP_DIR}"
echo "  Archives    : ${ARCHIVE_COUNT}"
echo "  Manifest    : ${MANIFEST}"
echo ""
echo "Archives created:"
for f in "${BACKUP_DIR}"/*.tar.gz; do
    if [[ -f "$f" ]]; then
        local_size="$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)"
        echo "  $(basename "$f")  $(( local_size / 1024 ))KB"
    fi
done
echo ""
echo "To restore: scripts/restore.sh $(basename "$BACKUP_DIR") --target <restore_root>"
