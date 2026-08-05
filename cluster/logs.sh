#!/bin/bash
# =============================================================================
#  cluster/logs.sh  --  read the command logs without remembering paths
# =============================================================================
#    bash cluster/logs.sh history [N]     last N history lines, aligned (def 30)
#    bash cluster/logs.sh running         runs with a START but no END
#    bash cluster/logs.sh failed [N]      only FAIL/ERROR lines
#    bash cluster/logs.sh list            every log file, newest first
#    bash cluster/logs.sh show <jobid>    everything recorded for one job
#    bash cluster/logs.sh tail            follow the newest log live
#    bash cluster/logs.sh grep <pattern>  search all logs
#    bash cluster/logs.sh stats           runs / failures / total time by context
#    bash cluster/logs.sh clean [DAYS]    delete trace+cli logs older than DAYS
#                                         (default 30; run_history.log is kept)
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Deliberately do NOT source config.sh here. config.sh sets PROJECT_DIR to a
# configured path that may point somewhere else entirely, and then this viewer
# would report "no logs yet" while sitting in a project full of them. This
# script lives in <project>/cluster/, so <project>/logs is unambiguous.
# An explicit $LOG_DIR still wins, for when you moved the logs deliberately.
LOG_DIR="${LOG_DIR:-$(dirname "${HERE}")/logs}"
HISTORY="${LOG_DIR}/run_history.log"

if [[ ! -d "${LOG_DIR}" ]]; then
    echo "No log directory yet: ${LOG_DIR}"
    echo "It is created the first time you run a job."
    exit 0
fi

# Render the tab-separated history into aligned columns.
_pretty() { column -t -s $'\t' 2>/dev/null || cat; }

case "${1:-history}" in

  history)
    N="${2:-30}"
    [[ -f "${HISTORY}" ]] || { echo "no history yet: ${HISTORY}"; exit 0; }
    echo "Last ${N} entries from ${HISTORY}:"
    echo
    tail -n "${N}" "${HISTORY}" | _pretty
    ;;

  running)
    [[ -f "${HISTORY}" ]] || { echo "no history yet"; exit 0; }
    echo "Runs with a START but no matching END/OK/FAIL"
    echo "(either still going, or killed hard enough to skip the exit trap):"
    echo
    # TWO passes over the file, not one. A single pass marks jobs done only as
    # it encounters their END line, so every START would look unfinished until
    # its END is read -- reporting completed runs as still running. Pass 1
    # collects all finished job IDs; pass 2 prints the STARTs not among them.
    # Keyed on job+context so a finished container step does not mask a still
    # running submit step of the same job.
    awk -F'\t' '
      FNR==NR {
        if ($2=="END") {
          k=""; c=""
          for(i=1;i<=NF;i++){ if($i ~ /^job=/) k=$i; if($i ~ /^ctx=/) c=$i }
          if (k!="") done[k SUBSEP c]=1
        }
        next
      }
      $2=="START" {
        k=""; c=""
        for(i=1;i<=NF;i++){ if($i ~ /^job=/) k=$i; if($i ~ /^ctx=/) c=$i }
        if (k!="" && !((k SUBSEP c) in done) && !seen[k SUBSEP c]++) print
      }
    ' "${HISTORY}" "${HISTORY}" | _pretty
    ;;

  failed)
    N="${2:-30}"
    [[ -f "${HISTORY}" ]] || { echo "no history yet"; exit 0; }
    # Catch both spellings of failure: an explicit FAIL/ERROR state (bash
    # log_cmd) and a terminal END carrying a non-zero rc (cli.py).
    awk -F'\t' '
      $2=="FAIL" || $2=="ERROR" { print; next }
      { for(i=1;i<=NF;i++) if($i ~ /^rc=/ && $i != "rc=0") { print; break } }
    ' "${HISTORY}" | tail -n "${N}" | _pretty
    if ! awk -F'\t' '$2=="FAIL" || $2=="ERROR" { f=1 }
                     { for(i=1;i<=NF;i++) if($i ~ /^rc=/ && $i != "rc=0") f=1 }
                     END { exit !f }' "${HISTORY}"; then
        echo "no failures recorded - good"
    fi
    ;;

  list)
    echo "Logs in ${LOG_DIR} (newest first):"
    echo
    ls -lht "${LOG_DIR}" | tail -n +2 | awk '{printf "  %-6s %s %s %s  %s\n", $5,$6,$7,$8,$9}'
    ;;

  show)
    JOB="${2:-}"
    [[ -z "${JOB}" ]] && { echo "usage: logs.sh show <jobid>" >&2; exit 1; }
    echo "=== history lines for job ${JOB} ==="
    grep -F "job=${JOB}" "${HISTORY}" 2>/dev/null | _pretty || echo "  (none)"
    echo
    for f in "${LOG_DIR}"/*"${JOB}"*.log; do
        [[ -e "$f" ]] || continue
        echo "=== ${f##*/} ($(wc -l < "$f") lines) ==="
        cat "$f"
        echo
    done
    ;;

  tail)
    NEWEST="$(ls -t "${LOG_DIR}"/*.log 2>/dev/null | head -1)"
    [[ -z "${NEWEST}" ]] && { echo "no log files yet"; exit 0; }
    echo "Following ${NEWEST}  (Ctrl-C to stop)"
    echo
    tail -f "${NEWEST}"
    ;;

  grep)
    PAT="${2:-}"
    [[ -z "${PAT}" ]] && { echo "usage: logs.sh grep <pattern>" >&2; exit 1; }
    grep -rn --color=auto "${PAT}" "${LOG_DIR}" || echo "no matches for '${PAT}'"
    ;;

  stats)
    [[ -f "${HISTORY}" ]] || { echo "no history yet"; exit 0; }
    awk -F'\t' '
      {
        ctx=""; rc=""; dur=0
        for (i=1;i<=NF;i++) {
          if ($i ~ /^ctx=/)  { ctx=substr($i,5) }
          if ($i ~ /^rc=/)   { rc=substr($i,4) }
          if ($i ~ /^dur=/)  { d=substr($i,5); sub(/s$/,"",d); dur=d+0 }
        }
        if (ctx=="") next
        if ($2=="START") starts[ctx]++
        # A run is counted once, at whichever terminal record it produced:
        # OK/FAIL from bash log_cmd, or END carrying rc= from a whole script
        # (cli.py and log_finish). Counting both would double-count each job.
        if ($2=="OK")   { ok[ctx]++;   total[ctx]+=dur }
        if ($2=="FAIL") { fail[ctx]++; total[ctx]+=dur }
        if ($2=="END")  { if (rc=="0" || rc=="") ok[ctx]++; else fail[ctx]++
                          total[ctx]+=dur }
        seen[ctx]=1
      }
      END {
        printf "%-22s %8s %8s %8s %12s\n", "CONTEXT","STARTED","OK","FAILED","TOTAL TIME"
        printf "%-22s %8s %8s %8s %12s\n", "----------------------","-------","--","------","----------"
        for (c in seen) {
          t=total[c]; h=int(t/3600); m=int((t%3600)/60); s=int(t%60)
          printf "%-22s %8d %8d %8d %9dh%02dm%02ds\n", c, starts[c], ok[c], fail[c], h, m, s
        }
      }
    ' "${HISTORY}"
    ;;

  clean)
    DAYS="${2:-30}"
    echo "Deleting trace_* and cli_* logs older than ${DAYS} days from ${LOG_DIR}"
    echo "(run_history.log is never deleted -- it is the permanent audit trail)"
    N=$(find "${LOG_DIR}" -maxdepth 1 -type f \
             \( -name 'trace_*.log' -o -name 'cli_*.log' -o -name '*.out' \) \
             -mtime "+${DAYS}" -print -delete | wc -l)
    echo "removed ${N} file(s)"
    ;;

  *)
    sed -n '3,18p' "$0"
    exit 1
    ;;
esac
