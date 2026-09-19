#!/usr/bin/env bash
#
# Is it alive and is it still collecting?
#
#     scalper-health
#
# Short enough to read on a phone over SSH, which is the point: the failure
# this guards against is not noticing for three days that nothing has been
# logged.

set -uo pipefail

APP_DIR="/home/scalper/btc-scalper"
LOG="${APP_DIR}/predictions.jsonl"
RUN="${APP_DIR}/runner.jsonl"

line() { printf '  %-22s %s\n' "$1" "$2"; }

echo
echo "  btc-scalper  $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "  ----------------------------------------------"

for svc in btc-logger btc-runner; do
  state=$(systemctl is-active "${svc}" 2>/dev/null || echo "unknown")
  since=$(systemctl show "${svc}" -p ActiveEnterTimestamp --value 2>/dev/null)
  restarts=$(systemctl show "${svc}" -p NRestarts --value 2>/dev/null)
  line "${svc}" "${state}  (restarts: ${restarts:-?})"
  [ -n "${since}" ] && line "" "up since ${since}"
done

echo "  ----------------------------------------------"

# The number that matters. A running service that stopped writing is the
# failure mode you would otherwise miss.
if [ -f "${LOG}" ]; then
  age=$(( ($(date +%s) - $(stat -c %Y "${LOG}")) / 60 ))
  windows=$(grep -c '"type": *"prediction"' "${LOG}" 2>/dev/null || echo 0)
  settled=$(grep -c '"type": *"settlement"' "${LOG}" 2>/dev/null || echo 0)
  line "last write" "${age} min ago"
  line "predictions" "${windows}"
  line "settlements" "${settled}"
  if [ "${age}" -gt 20 ]; then
    echo
    echo "  WARNING: nothing written for ${age} minutes."
    echo "  A window closes every 15. Check:  journalctl -u btc-logger -n 40"
  fi
else
  line "predictions.jsonl" "MISSING"
fi

if [ -f "${RUN}" ]; then
  posted=$(grep -c '"action": *"posted"' "${RUN}" 2>/dev/null || echo 0)
  line "orders posted" "${posted}"
fi

if [ -f "${APP_DIR}/KILL" ]; then
  echo
  echo "  KILL SWITCH IS SET - the runner will place nothing."
  echo "  reason: $(head -1 "${APP_DIR}/KILL")"
  echo "  clear with: rm ${APP_DIR}/KILL"
fi

echo
echo "  score:  cd ${APP_DIR} && .venv/bin/python score.py"
echo "  fills:  cd ${APP_DIR} && .venv/bin/python makerstats.py"
echo
