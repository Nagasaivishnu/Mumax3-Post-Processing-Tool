#!/bin/bash
# =============================================================================
#  cluster/logging.sh  --  shared command logging for every script here
# =============================================================================
#  Sourced by the SLURM scripts, container_script.sh, interactive.sh and
#  setup_venv.sh. Produces two things:
#
#  1. AUDIT TRAIL  --  logs/run_history.log
#     One tab-separated line per command, appended forever. Grep-friendly:
#
#       2026-08-04T20:41:07+08:00  START  job=4471  host=blade07  user=e1234567
#                                  ctx=submit_modes  cmd=singularity --exec ...
#       2026-08-04T21:12:53+08:00  OK     job=4471  host=blade07  user=e1234567
#                                  ctx=submit_modes  rc=0  dur=1906s
#                                  cmd=singularity --exec ...
#
#     Writes are made with `flock` where available, so concurrent array tasks
#     appending to the same file cannot interleave mid-line.
#
#  2. FULL TRACE  --  logs/trace_<context>_<jobid>.log
#     Every command the script executes, via bash xtrace with a timestamped
#     PS4. This is the "what exactly ran, in what order" record.
#
#  Usage in a script:
#
#      source "${HERE}/logging.sh"
#      log_init submit_modes            # sets up both files
#      log_info "staging input"
#      log_cmd rsync -a "$SRC/" "$DST/" # times it, records rc, still runs it
#      log_finish $?
#
#  Environment overrides:
#      LOG_DIR      where logs go            (default: <project>/logs)
#      LOG_TRACE=0  disable the xtrace file  (audit trail still written)
#      LOG_QUIET=1  suppress console echo    (files still written)
# =============================================================================

# Guard against double-sourcing (interactive.sh -> container_script.sh).
[[ -n "${_MUMAX_LOGGING_SOURCED:-}" ]] && return 0
_MUMAX_LOGGING_SOURCED=1

# --- Where ------------------------------------------------------------------
# Default to <project>/logs. config.sh may already have set PROJECT_DIR; if not,
# fall back to the parent of this file's directory.
_LOG_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR:-$(dirname "${_LOG_SELF_DIR}")}/logs}"

LOG_CONTEXT="${LOG_CONTEXT:-mumax}"
LOG_HISTORY=""
LOG_TRACE_FILE=""
_LOG_T0=0

# --- Timestamp helper --------------------------------------------------------
# ISO-8601 with timezone, so logs from different sessions sort correctly and
# you can tell SGT apart from UTC when comparing against SLURM's own records.
log_ts() { date +"%Y-%m-%dT%H:%M:%S%:z"; }

# --- Internal: pause/resume xtrace ------------------------------------------
#  Without this, `set -x` traces the logging functions themselves and the trace
#  file fills up with printf/flock noise instead of the commands you care about.
#  Written as one-liners on purpose: bash traces each simple command, so the
#  fewer commands before `set +x` takes effect, the less noise leaks through.
_log_noxtrace() { [[ $- == *x* ]] && { _LOG_X=1; set +x; } || _LOG_X=0; return 0; }
_log_rextrace() { [[ "${_LOG_X:-0}" == 1 ]] && set -x; return 0; }

# --- Internal: append one line atomically ------------------------------------
_log_append() {
    _log_noxtrace
    local line="$1"
    if [[ -n "${LOG_HISTORY}" ]]; then
        if command -v flock >/dev/null 2>&1; then
            # Concurrent array tasks share run_history.log; flock keeps lines whole.
            ( flock -x 200; printf '%s\n' "${line}" >&200 ) 200>>"${LOG_HISTORY}"
        else
            printf '%s\n' "${line}" >> "${LOG_HISTORY}"
        fi
    fi
    _log_rextrace
}

_log_fields() {
    printf 'job=%s\thost=%s\tuser=%s\tctx=%s' \
        "${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+.${SLURM_ARRAY_TASK_ID}}" \
        "$(hostname -s 2>/dev/null || echo unknown)" \
        "${USER:-unknown}" \
        "${LOG_CONTEXT}"
}

# --- Public: initialise ------------------------------------------------------
#  log_init <context> [the script's own "$@"]
log_init() {
    LOG_CONTEXT="${1:-${LOG_CONTEXT}}"
    shift || true
    local argv="$*"
    _LOG_T0=$SECONDS

    mkdir -p "${LOG_DIR}" 2>/dev/null || {
        echo "WARNING: cannot create ${LOG_DIR}; logging to /tmp instead" >&2
        LOG_DIR="/tmp/mumax_logs_${USER:-nobody}"
        mkdir -p "${LOG_DIR}"
    }

    LOG_HISTORY="${LOG_DIR}/run_history.log"
    touch "${LOG_HISTORY}" 2>/dev/null || LOG_HISTORY=""

    local tag="${SLURM_JOB_ID:-$$}${SLURM_ARRAY_TASK_ID:+_${SLURM_ARRAY_TASK_ID}}"

    # --- full command trace --------------------------------------------------
    if [[ "${LOG_TRACE:-1}" != "0" ]]; then
        LOG_TRACE_FILE="${LOG_DIR}/trace_${LOG_CONTEXT}_${tag}.log"
        # PS4 is printed before every traced command. \D{} is bash's built-in
        # prompt strftime -- far cheaper than forking `date` on every line, which
        # matters because xtrace fires thousands of times per job.
        export PS4='+ \D{%H:%M:%S} ${BASH_SOURCE##*/}:${LINENO}: '
        # fd 9 keeps the trace out of stdout, so it does not pollute the SLURM
        # .out file or get mixed into command output.
        exec 9>>"${LOG_TRACE_FILE}"
        BASH_XTRACEFD=9
        export BASH_XTRACEFD
        set -x
    fi

    log_info "=== ${LOG_CONTEXT} started ==="
    log_info "invocation: ${0##*/} ${argv}"
    log_info "logs: history=${LOG_HISTORY:-disabled} trace=${LOG_TRACE_FILE:-disabled}"
    _log_append "$(log_ts)"$'\t'"START"$'\t'"$(_log_fields)"$'\t'"cmd=${0##*/} ${argv}"
}

# --- Public: human-readable message -----------------------------------------
log_info() {
    _log_noxtrace
    local line="[$(log_ts)] [${LOG_CONTEXT}] $*"
    [[ "${LOG_QUIET:-0}" == "1" ]] || echo "${line}"
    [[ -n "${LOG_TRACE_FILE}" ]] && echo "${line}" >&9 2>/dev/null
    _log_rextrace
}

log_warn() { log_info "WARNING: $*" >&2; }

log_error() {
    _log_noxtrace
    echo "[$(log_ts)] [${LOG_CONTEXT}] ERROR: $*" >&2
    _log_rextrace
    _log_append "$(log_ts)"$'\t'"ERROR"$'\t'"$(_log_fields)"$'\t'"msg=$*"
}

# --- Public: run a command, timed and recorded -------------------------------
#  Runs "$@" verbatim. Records START before and OK/FAIL after, with the exit
#  code and duration. Returns the command's own exit status, so callers can
#  still branch on it. Deliberately does NOT swallow failures.
log_cmd() {
    local cmd_str="$*"
    local start_s=$SECONDS
    local rc=0

    log_info "RUN: ${cmd_str}"
    _log_append "$(log_ts)"$'\t'"START"$'\t'"$(_log_fields)"$'\t'"cmd=${cmd_str}"

    # Temporarily relax errexit so we always get to record the result, then
    # hand the status back to the caller unchanged.
    local had_e=0
    [[ $- == *e* ]] && had_e=1 && set +e
    "$@"
    rc=$?
    [[ $had_e == 1 ]] && set -e

    local dur=$((SECONDS - start_s))
    if [[ $rc -eq 0 ]]; then
        log_info "OK  (${dur}s): ${cmd_str}"
        _log_append "$(log_ts)"$'\t'"OK"$'\t'"$(_log_fields)"$'\t'"rc=0"$'\t'"dur=${dur}s"$'\t'"cmd=${cmd_str}"
    else
        log_info "FAIL rc=${rc} (${dur}s): ${cmd_str}"
        _log_append "$(log_ts)"$'\t'"FAIL"$'\t'"$(_log_fields)"$'\t'"rc=${rc}"$'\t'"dur=${dur}s"$'\t'"cmd=${cmd_str}"
    fi
    return $rc
}

# --- Public: close out -------------------------------------------------------
log_finish() {
    local rc="${1:-0}"
    local dur=$((SECONDS - _LOG_T0))
    local state="OK"
    [[ "${rc}" -ne 0 ]] && state="FAIL"

    log_info "=== ${LOG_CONTEXT} finished rc=${rc} after ${dur}s ==="
    _log_append "$(log_ts)"$'\t'"END"$'\t'"$(_log_fields)"$'\t'"rc=${rc}"$'\t'"dur=${dur}s"$'\t'"cmd=${0##*/}"

    if [[ -n "${LOG_TRACE_FILE}" ]]; then
        set +x
        exec 9>&- 2>/dev/null || true

        # Strip the logging library's own trace lines. bash traces every simple
        # command, including the ones inside log_info/log_cmd, and without this
        # ~75% of the trace is logging.sh bookkeeping rather than the commands
        # you actually want to see. Done once here rather than by filtering the
        # trace fd live, which would race with the script exiting.
        # PS4 puts "<file>:<line>:" on every trace line, so this is exact.
        if [[ -f "${LOG_TRACE_FILE}" ]] && command -v sed >/dev/null 2>&1; then
            local kept before after
            before=$(wc -l < "${LOG_TRACE_FILE}" 2>/dev/null || echo 0)
            sed -i '/ logging\.sh:[0-9]\+:/d' "${LOG_TRACE_FILE}" 2>/dev/null || true
            after=$(wc -l < "${LOG_TRACE_FILE}" 2>/dev/null || echo 0)
            kept=$((before - after))
            [[ ${kept} -gt 0 ]] && echo "Trace: dropped ${kept} logging-internal lines, kept ${after}"
        fi

        echo "Full command trace: ${LOG_TRACE_FILE}"
    fi
    [[ -n "${LOG_HISTORY}" ]] && echo "Run history:        ${LOG_HISTORY}"
    return "${rc}"
}

# --- Public: make sure log_finish runs even on failure or scancel -------------
#  Call once, right after log_init, in scripts that use `set -e` or that a user
#  might cancel. Without this, a `scancel` leaves no END record and the run
#  looks like it is still going when you read the history later.
log_trap_exit() {
    trap 'rc=$?; log_finish $rc >/dev/null 2>&1 || true' EXIT
    trap 'log_error "received SIGTERM (scancel or walltime hit)"; exit 143' TERM
    trap 'log_error "received SIGINT"; exit 130' INT
}
