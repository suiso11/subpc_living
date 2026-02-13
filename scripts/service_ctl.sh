#!/bin/bash
# =============================================================================
# subpc_living サービス管理ヘルパー
# systemctl コマンドをラップし、サービス名を覚えなくてよくする
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ユーザーサービス一覧
USER_SERVICES=("subpc-web" "subpc-voice")
# システムサービス一覧
SYSTEM_SERVICES=("subpc-gpu-powersave")

COLOR_GREEN="\033[92m"
COLOR_RED="\033[91m"
COLOR_YELLOW="\033[93m"
COLOR_CYAN="\033[96m"
COLOR_DIM="\033[2m"
COLOR_BOLD="\033[1m"
COLOR_RESET="\033[0m"

usage() {
    echo ""
    echo -e "${COLOR_CYAN}${COLOR_BOLD}subpc_living サービス管理${COLOR_RESET}"
    echo ""
    echo "使い方:"
    echo "  bash scripts/service_ctl.sh <コマンド> [サービス名]"
    echo ""
    echo "コマンド:"
    echo "  status     全サービスの状態を表示"
    echo "  start      サービスを開始 (デフォルト: web)"
    echo "  stop       サービスを停止"
    echo "  restart    サービスを再起動"
    echo "  enable     サービスの自動起動を有効化"
    echo "  disable    サービスの自動起動を無効化"
    echo "  logs       サービスのログを表示 (-f でフォロー)"
    echo "  health     ヘルスチェックを実行"
    echo "  gpu        GPU の状態を表示"
    echo ""
    echo "サービス名:"
    echo "  web        Web UI サーバー (subpc-web)"
    echo "  voice      音声対話パイプライン (subpc-voice)"
    echo "  gpu        GPU 省電力制御 (subpc-gpu-powersave, 要 sudo)"
    echo "  all        全サービス"
    echo ""
    echo "例:"
    echo "  bash scripts/service_ctl.sh status"
    echo "  bash scripts/service_ctl.sh start web"
    echo "  bash scripts/service_ctl.sh logs web"
    echo "  bash scripts/service_ctl.sh logs web -f"
    echo ""
}

resolve_service() {
    case "$1" in
        web)   echo "subpc-web" ;;
        voice) echo "subpc-voice" ;;
        gpu)   echo "subpc-gpu-powersave" ;;
        *)     echo "$1" ;;
    esac
}

is_system_service() {
    local svc="$1"
    for s in "${SYSTEM_SERVICES[@]}"; do
        [ "$svc" = "$s" ] && return 0
    done
    return 1
}

svc_ctl() {
    local svc="$1"
    shift
    if is_system_service "$svc"; then
        sudo systemctl "$@" "$svc"
    else
        systemctl --user "$@" "$svc"
    fi
}

show_status() {
    echo ""
    echo -e "${COLOR_CYAN}${COLOR_BOLD}=== ユーザーサービス ===${COLOR_RESET}"
    for svc in "${USER_SERVICES[@]}"; do
        local state
        state=$(systemctl --user is-active "$svc" 2>/dev/null || echo "unknown")
        local enabled
        enabled=$(systemctl --user is-enabled "$svc" 2>/dev/null || echo "unknown")
        local icon
        case "$state" in
            active)   icon="${COLOR_GREEN}● active${COLOR_RESET}" ;;
            inactive) icon="${COLOR_DIM}○ inactive${COLOR_RESET}" ;;
            failed)   icon="${COLOR_RED}✕ failed${COLOR_RESET}" ;;
            *)        icon="${COLOR_YELLOW}? ${state}${COLOR_RESET}" ;;
        esac
        echo -e "  ${svc}: ${icon}  (enabled: ${enabled})"
    done

    echo ""
    echo -e "${COLOR_CYAN}${COLOR_BOLD}=== システムサービス ===${COLOR_RESET}"
    for svc in "${SYSTEM_SERVICES[@]}"; do
        local state
        state=$(systemctl is-active "$svc" 2>/dev/null || echo "unknown")
        local enabled
        enabled=$(systemctl is-enabled "$svc" 2>/dev/null || echo "unknown")
        local icon
        case "$state" in
            active)   icon="${COLOR_GREEN}● active${COLOR_RESET}" ;;
            inactive) icon="${COLOR_DIM}○ inactive${COLOR_RESET}" ;;
            failed)   icon="${COLOR_RED}✕ failed${COLOR_RESET}" ;;
            *)        icon="${COLOR_YELLOW}? ${state}${COLOR_RESET}" ;;
        esac
        echo -e "  ${svc}: ${icon}  (enabled: ${enabled})"
    done

    echo ""
    echo -e "${COLOR_CYAN}${COLOR_BOLD}=== 関連サービス ===${COLOR_RESET}"
    local ollama_state
    ollama_state=$(systemctl is-active ollama 2>/dev/null || echo "unknown")
    local ollama_icon
    case "$ollama_state" in
        active) ollama_icon="${COLOR_GREEN}● active${COLOR_RESET}" ;;
        *)      ollama_icon="${COLOR_YELLOW}? ${ollama_state}${COLOR_RESET}" ;;
    esac
    echo -e "  ollama: ${ollama_icon}"
    echo ""
}

cmd_start() {
    local target="${1:-web}"
    if [ "$target" = "all" ]; then
        for svc in "${USER_SERVICES[@]}"; do
            echo -e "Starting ${COLOR_CYAN}${svc}${COLOR_RESET}..."
            svc_ctl "$svc" start
        done
    else
        local svc
        svc=$(resolve_service "$target")
        echo -e "Starting ${COLOR_CYAN}${svc}${COLOR_RESET}..."
        svc_ctl "$svc" start
    fi
}

cmd_stop() {
    local target="${1:-web}"
    if [ "$target" = "all" ]; then
        for svc in "${USER_SERVICES[@]}"; do
            echo -e "Stopping ${COLOR_CYAN}${svc}${COLOR_RESET}..."
            svc_ctl "$svc" stop
        done
    else
        local svc
        svc=$(resolve_service "$target")
        echo -e "Stopping ${COLOR_CYAN}${svc}${COLOR_RESET}..."
        svc_ctl "$svc" stop
    fi
}

cmd_restart() {
    local target="${1:-web}"
    if [ "$target" = "all" ]; then
        for svc in "${USER_SERVICES[@]}"; do
            echo -e "Restarting ${COLOR_CYAN}${svc}${COLOR_RESET}..."
            svc_ctl "$svc" restart
        done
    else
        local svc
        svc=$(resolve_service "$target")
        echo -e "Restarting ${COLOR_CYAN}${svc}${COLOR_RESET}..."
        svc_ctl "$svc" restart
    fi
}

cmd_enable() {
    local target="${1:-web}"
    if [ "$target" = "all" ]; then
        for svc in "${USER_SERVICES[@]}"; do
            echo -e "Enabling ${COLOR_CYAN}${svc}${COLOR_RESET}..."
            svc_ctl "$svc" enable
        done
    else
        local svc
        svc=$(resolve_service "$target")
        echo -e "Enabling ${COLOR_CYAN}${svc}${COLOR_RESET}..."
        svc_ctl "$svc" enable
    fi
}

cmd_disable() {
    local target="${1:-web}"
    if [ "$target" = "all" ]; then
        for svc in "${USER_SERVICES[@]}"; do
            echo -e "Disabling ${COLOR_CYAN}${svc}${COLOR_RESET}..."
            svc_ctl "$svc" disable
        done
    else
        local svc
        svc=$(resolve_service "$target")
        echo -e "Disabling ${COLOR_CYAN}${svc}${COLOR_RESET}..."
        svc_ctl "$svc" disable
    fi
}

cmd_logs() {
    local target="${1:-web}"
    shift 2>/dev/null || true
    local svc
    svc=$(resolve_service "$target")

    if is_system_service "$svc"; then
        journalctl -u "$svc" "$@"
    else
        journalctl --user -u "$svc" "$@"
    fi
}

cmd_health() {
    source "${PROJECT_ROOT}/.venv/bin/activate"
    python -m src.service.healthcheck
}

cmd_gpu() {
    source "${PROJECT_ROOT}/.venv/bin/activate"
    python -m src.service.power
}

# --- メイン ---
CMD="${1:-}"
shift 2>/dev/null || true

case "$CMD" in
    status)  show_status ;;
    start)   cmd_start "$@" ;;
    stop)    cmd_stop "$@" ;;
    restart) cmd_restart "$@" ;;
    enable)  cmd_enable "$@" ;;
    disable) cmd_disable "$@" ;;
    logs)    cmd_logs "$@" ;;
    health)  cmd_health ;;
    gpu)     cmd_gpu ;;
    -h|--help|help|"")
        usage ;;
    *)
        echo "Unknown command: ${CMD}"
        usage
        exit 1
        ;;
esac
