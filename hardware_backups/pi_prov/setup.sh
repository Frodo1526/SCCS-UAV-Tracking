#!/bin/bash

# When LOG_FILE is set, info/warn/error also append a plain timestamped line to it.
LOG_FILE=
_log() { if [[ -n "$LOG_FILE" ]]; then printf '%s [%s] %s\n' "$(date '+%F %T')" "$1" "$2" >> "$LOG_FILE"; fi; }

info()  { printf '\033[1;34m[INFO]\033[0m %s\n' "$1" >&2; _log INFO "$1"; }
warn()  { printf '\033[1;33m[WARN]\033[0m %s\n' "$1" >&2; _log WARN "$1"; }
error() { printf '\033[1;31m[ERROR]\033[0m %s\n' "$1" >&2; _log ERROR "$1"; }

CONFIG="${CONFIG:-/boot/firmware/config.txt}"   # override for testing: CONFIG=~/config.test ./setup.sh --dryrun
VERIFY_LOG="${VERIFY_LOG:-/var/log/flight_logs/setup_verify.log}"   # dir is created by the service's LogsDirectory=
depts=(python3 python3-picamera2 rpicam-apps gstreamer1.0-libcamera gstreamer1.0-plugins-ugly python3-gi openssh-server hailo-all)

set -uo pipefail

usage() {
    echo "usage: $0 --setup | --verify | --dryrun" >&2
}

# Applies the config.txt edits to the file passed in ($1).
# Uses sudo only when that file isn't writable by the current user.
edit_config() {
    local f="$1"
    local SUDO=sudo
    [[ -w "$f" ]] && SUDO=

    if grep -q '^camera_auto_detect=' "$f"; then
        $SUDO sed -i 's/^camera_auto_detect=.*/camera_auto_detect=0/' "$f"
    else
        echo 'camera_auto_detect=0' | $SUDO tee -a "$f" >/dev/null
    fi

    if ! grep -qx 'dtoverlay=imx477' "$f"; then
        printf '\n[all]\ndtoverlay=imx477\n' | $SUDO tee -a "$f" >/dev/null
    fi

    $SUDO sed -i 's/^dtoverlay=arducam-pivariety/#&/' "$f"
    $SUDO sed -i 's/^dtoverlay=imx477,cam0/#&/' "$f"
}

do_setup() {
    info "sudo elevation is required for updating and installing packages"
    trap 'sudo -k' EXIT

    sudo apt-get update && sudo apt-get upgrade -y
    sudo apt-get install -y "${depts[@]}"

    sudo cp "$CONFIG" "$CONFIG.bak.$(date +%Y%m%d-%H%M%S)"
    edit_config "$CONFIG"
}

do_verify() {
    local fails=0

    if touch "$VERIFY_LOG" 2>/dev/null; then
        LOG_FILE="$VERIFY_LOG"
    else
        warn "can't write $VERIFY_LOG, verify results will only be shown on screen"
    fi
    info "verify: started on $(hostname)"

    grep -qx 'camera_auto_detect=0'            "$CONFIG" || { error "camera_auto_detect not 0"; ((fails++)); }
    grep -qx 'dtoverlay=imx477'                "$CONFIG" || { error "imx477 overlay missing";   ((fails++)); }
    grep -q  '^dtoverlay=arducam-pivariety'    "$CONFIG" && { error "stale pivariety overlay";  ((fails++)); }

    if ! rpicam-hello --list-cameras 2>&1 | grep -q "imx477"; then
        error "no camera detected! please ensure the camera module is correctly mounted"
        ((fails++))
    fi

    if ((fails)); then
        error "verify: $fails check(s) failed"
        return 1
    fi
    info "verify: all checks passed"
}

do_dryrun() {
    info "packages that would be installed:"
    apt-get install -s "${depts[@]}" | grep '^Inst' || info "  (none, all installed)"

    local tmp
    tmp=$(mktemp)
    cp "$CONFIG" "$tmp"
    edit_config "$tmp"
    info "config.txt changes:"
    diff -u "$CONFIG" "$tmp" || true
    rm -f "$tmp"
}

case "${1:-}" in
    --setup)   do_setup ;;
    --verify)  do_verify ;;
    --dryrun)  do_dryrun ;;
    -h|--help) usage; exit 0 ;;
    *)         usage; exit 2 ;;
esac
