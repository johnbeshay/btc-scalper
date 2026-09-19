#!/usr/bin/env bash
#
# Set up a fresh Ubuntu box to run the logger and the maker runner
# continuously. Run as root on a brand new server:
#
#     bash setup.sh
#
# Idempotent - safe to run again after a change.
#
# WHY A SERVER AT ALL
# -------------------
# The logger needs weeks of uninterrupted windows and a laptop that moves
# around does not provide that. Two deaths and one network drop already cost
# windows that cannot be recovered: Kalshi does not serve historical order
# books, so a gap in the log is a permanent hole in the evidence.

set -euo pipefail

REPO="https://github.com/johnbeshay/btc-scalper.git"
USER_NAME="scalper"
HOME_DIR="/home/${USER_NAME}"
APP_DIR="${HOME_DIR}/btc-scalper"

echo "==> packages"
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip git chrony curl

# The Kalshi signature covers the millisecond timestamp, so a drifting clock
# produces 401s that look exactly like bad credentials. This cost an hour of
# debugging once already; chrony makes it not happen.
echo "==> clock sync"
systemctl enable --now chrony
sleep 2
timedatectl set-ntp true || true
timedatectl | sed -n 's/^ *//p' | grep -E 'Time zone|synchronized|NTP' || true

echo "==> user"
id -u "${USER_NAME}" >/dev/null 2>&1 || useradd -m -s /bin/bash "${USER_NAME}"

echo "==> code"
if [ -d "${APP_DIR}/.git" ]; then
  sudo -u "${USER_NAME}" git -C "${APP_DIR}" pull --ff-only
else
  sudo -u "${USER_NAME}" git clone "${REPO}" "${APP_DIR}"
fi

echo "==> python environment"
sudo -u "${USER_NAME}" python3 -m venv "${APP_DIR}/.venv"
sudo -u "${USER_NAME}" "${APP_DIR}/.venv/bin/pip" install -q --upgrade pip
# cryptography is the only dependency, and only the auth module imports it.
sudo -u "${USER_NAME}" "${APP_DIR}/.venv/bin/pip" install -q cryptography

sudo -u "${USER_NAME}" mkdir -p "${APP_DIR}/logs" "${HOME_DIR}/backups"

echo "==> tests"
if sudo -u "${USER_NAME}" bash -c "cd ${APP_DIR} && .venv/bin/python -m unittest discover -p 'test_*.py' -q" 2>&1 | tail -3; then
  echo "    tests pass"
else
  echo "    TESTS FAILED - fix before starting anything"
  exit 1
fi

echo "==> services"
install -m 644 "${APP_DIR}/deploy/btc-logger.service" /etc/systemd/system/
install -m 644 "${APP_DIR}/deploy/btc-runner.service" /etc/systemd/system/
install -m 755 "${APP_DIR}/deploy/health.sh" /usr/local/bin/scalper-health
systemctl daemon-reload

# Nightly copy of predictions.jsonl. It is the only record of the evidence
# and nothing else can recreate it.
cat > /etc/cron.d/btc-scalper-backup <<EOF
17 3 * * * ${USER_NAME} cp ${APP_DIR}/predictions.jsonl ${HOME_DIR}/backups/predictions-\$(date +\%Y\%m\%d).jsonl 2>/dev/null; find ${HOME_DIR}/backups -name 'predictions-*.jsonl' -mtime +14 -delete
EOF

echo
echo "================================================================"
echo "  Setup done. NOT started yet - credentials are still missing."
echo "================================================================"
echo
echo "  From your laptop, copy the two secrets across:"
echo
echo "    scp kalshi-demo-credentials.json root@THIS_SERVER:${APP_DIR}/"
echo "    scp kalshi-demo.key            root@THIS_SERVER:${APP_DIR}/"
echo
echo "  Then back here:"
echo
echo "    chown ${USER_NAME}:${USER_NAME} ${APP_DIR}/kalshi-demo*"
echo "    chmod 600 ${APP_DIR}/kalshi-demo.key"
echo "    sudo -u ${USER_NAME} ${APP_DIR}/.venv/bin/python ${APP_DIR}/executor.py check"
echo
echo "  If that prints a balance, start both services:"
echo
echo "    systemctl enable --now btc-logger btc-runner"
echo "    scalper-health"
echo
