# Running this on a server

The logger needs weeks of uninterrupted windows. A laptop that moves around
does not provide that — it has already died twice and dropped its network
once, and every gap is permanent: Kalshi does not serve historical order
books, so a missing window can never be recovered.

About $5/month fixes it.

---

## 1. Get a server

Any small Ubuntu box. Cheapest reasonable options:

| provider | plan | cost |
|---|---|---|
| Hetzner | CX22 | ~€4/mo |
| DigitalOcean | Basic droplet | $6/mo |
| Vultr | Regular | $5/mo |

Choose **Ubuntu 24.04**, the smallest size, and a region near you. Add your
SSH key during creation if it offers — otherwise it emails a root password.

This workload is tiny: two Python processes making a few HTTP calls a
minute.

## 2. Connect

```bash
ssh root@YOUR_SERVER_IP
```

From Windows, PowerShell has `ssh` built in.

## 3. Run the setup script

```bash
curl -fsSL https://raw.githubusercontent.com/johnbeshay/btc-scalper/main/deploy/setup.sh -o setup.sh
less setup.sh          # read it before running it
bash setup.sh
```

It installs Python and git, creates a `scalper` user, clones the repo, makes
a virtualenv with `cryptography`, runs the test suite, installs the two
systemd services, and sets up a nightly backup of `predictions.jsonl`.

**It also installs `chrony` for clock sync.** That is not incidental: Kalshi
signs the millisecond timestamp with every request, so a drifting clock
produces 401s that look exactly like bad credentials.

It stops short of starting anything, because the credentials are not there
yet.

## 4. Copy the credentials across

These are gitignored, so the clone did not bring them. From **your laptop**,
in the repo directory:

```powershell
scp kalshi-demo-credentials.json root@YOUR_SERVER_IP:/home/scalper/btc-scalper/
scp kalshi-demo.key root@YOUR_SERVER_IP:/home/scalper/btc-scalper/
```

Back on the server:

```bash
chown scalper:scalper /home/scalper/btc-scalper/kalshi-demo*
chmod 600 /home/scalper/btc-scalper/kalshi-demo.key
```

## 5. Prove auth works before starting anything

```bash
sudo -u scalper /home/scalper/btc-scalper/.venv/bin/python \
  /home/scalper/btc-scalper/executor.py check
```

A balance means you are ready. A 401 means the clock or the key — check
`timedatectl` first, it is the more common cause.

## 6. Start

```bash
systemctl enable --now btc-logger btc-runner
scalper-health
```

`enable` means they come back after a reboot. `--now` starts them
immediately.

---

## Checking on it

One command, short enough to read on a phone:

```bash
scalper-health
```

It shows whether both services are running, how many times they have
restarted, **how long since anything was written**, how many windows and
settlements are logged, and whether the kill switch is set.

The "last write" line is the one that matters. A service can be running and
silently not collecting; a window closes every 15 minutes, so anything over
20 means something is wrong.

When you want the actual numbers:

```bash
cd /home/scalper/btc-scalper
.venv/bin/python score.py        # is there an edge
.venv/bin/python makerstats.py   # do resting orders fill
```

## If something looks wrong

```bash
journalctl -u btc-logger -n 50 --no-pager
journalctl -u btc-runner -n 50 --no-pager
systemctl restart btc-logger
```

## Stopping the runner immediately

```bash
touch /home/scalper/btc-scalper/KILL
```

The rails refuse every order while that file exists. It works even if the
process is wedged, and it is deliberately the crudest control in the
project. Remove the file to resume.

## Updating after a code change

```bash
cd /home/scalper/btc-scalper
sudo -u scalper git pull
sudo -u scalper .venv/bin/python -m unittest discover -p 'test_*.py'
systemctl restart btc-logger btc-runner
```

Run the tests before the restart, not after.

---

## Things worth knowing

**The restart policy is deliberate.** `Restart=always` with
`StartLimitIntervalSec=0` means systemd never gives up. The default would
stop retrying after five failures in ten seconds, which turns a brief
network outage into a silent permanent stop — precisely the failure this
deployment exists to prevent.

**`predictions.jsonl` is backed up nightly** to `/home/scalper/backups`,
kept for two weeks. It is the only record of the evidence and nothing can
recreate it. Consider pulling a copy to your laptop occasionally too:

```powershell
scp root@YOUR_SERVER_IP:/home/scalper/btc-scalper/predictions.jsonl .
```

**The runner is demo-only** while `BASE` in `core/kalshi_exec.py` points at
the demo host, and demo credentials cannot authenticate against production
regardless. Two independent barriers, and moving to a server does not weaken
either.

**Log files grow.** `logs/*.out` will accumulate; truncate them occasionally
or add logrotate if it ever matters. `predictions.jsonl` itself grows by
roughly a megabyte a week, which is nothing.
