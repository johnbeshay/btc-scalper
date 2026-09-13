"""
Offline test harness. Generates synthetic candles so the agents can be
exercised without hitting an exchange.
"""
import random
from datetime import datetime, timedelta, timezone

from core.feed import Candle
from core.indicators import FeeModel
from core.orchestrator import Orchestrator


def synth(n=120, start=43000.0, drift=0.0, vol=0.0015, seed=7):
    random.seed(seed)
    candles, price = [], start
    t = datetime.now(timezone.utc) - timedelta(minutes=5 * n)
    for i in range(n):
        o = price
        move = random.gauss(drift, vol)
        c = o * (1 + move)
        h = max(o, c) * (1 + abs(random.gauss(0, vol / 2)))
        l = min(o, c) * (1 - abs(random.gauss(0, vol / 2)))
        v = abs(random.gauss(120, 40))
        if i == n - 1:
            v *= 3.2
        candles.append(Candle(t + timedelta(minutes=5 * i), o, h, l, c, v))
        price = c
    return candles


if __name__ == "__main__":
    fees = FeeModel(entry_fee_pct=0.60, exit_fee_pct=0.60, slippage_pct=0.05)
    orch = Orchestrator(fees=fees)

    for label, kwargs in [
        ("TRENDING UP", dict(drift=0.0012, vol=0.0020, seed=3)),
        ("DEAD QUIET", dict(drift=0.0000, vol=0.0002, seed=5)),
        ("SELLING OFF", dict(drift=-0.0015, vol=0.0025, seed=11)),
    ]:
        print(f"\n\n########  {label}  ########")
        print(orch.describe(orch.run(synth(**kwargs))))
