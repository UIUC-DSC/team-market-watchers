"""
Build a CSV of wallets and their betting track records.

Wallets come from:
  - the leaderboard (all-time, this month, this week), ranked by profit and by volume
  - the top holders of the highest-volume open markets
  - the top holders of the highest-volume resolved markets
  - whoever made the most recent trades across all of Polymarket

The volume leaderboard and recent traders bring in ordinary and losing users,
not just top earners, so the dataset has something to compare the winners to.

For each wallet we pull its resolved bets (recent closed positions plus lost
bets it never redeemed) and compute win rate, profit, and how that win rate
compares to the prices the wallet paid.

Run with:
    .venv\\Scripts\\python build_wallet_dataset.py
    .venv\\Scripts\\python build_wallet_dataset.py --leaderboard 500 --markets 100 --max-closed 500

Reruns resume: wallets already in the output CSV are skipped. Pass --fresh to start over.

Wallets whose history hit --max-closed/--max-open are marked history_truncated.
To refetch just those with a higher cap (0 = no cap) and replace their rows:
    .venv\\Scripts\\python build_wallet_dataset.py --refetch-truncated --max-closed 1000 --out data/wallets_v2.csv
"""

import argparse
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from polymarket_api import PolymarketClient

_local = threading.local()

# Fixed column order, so every batch appended to the CSV lines up with the header
# (wallets with no bets return fewer fields).
COLUMNS = [
    "wallet", "name", "sources", "n_bets", "n_closed", "n_unredeemed", "n_open_unresolved",
    "wins", "losses", "win_rate", "avg_entry_price", "win_rate_edge", "total_pnl", "total_cost",
    "roi", "median_pnl", "first_end_date", "last_end_date", "history_truncated",
]
# n_closed was called n_redeemed in older CSVs. It counts every closed position:
# cashed in after the market resolved, or fully sold before it did.
OLD_NAMES = {"n_redeemed": "n_closed"}


def has_rows(path):
    """True if the CSV exists and isn't empty (a crash or restart can leave a 0-byte file)."""
    return os.path.exists(path) and os.path.getsize(path) > 0


def get_client():
    """One client (and HTTP session) per worker thread."""
    if not hasattr(_local, "client"):
        _local.client = PolymarketClient()
    return _local.client


def collect_wallets(client, n_leaderboard, n_markets, holders_per_outcome, n_closed_markets, n_recent_trades):
    """Return {wallet: {"name": ..., "sources": set(...)}} from every wallet source."""
    wallets = {}

    def add(wallet, name, source):
        entry = wallets.setdefault(wallet.lower(), {"name": name, "sources": set()})
        entry["sources"].add(source)
        if not entry["name"] and name:
            entry["name"] = name

    def add_holders(markets, source):
        for i, m in enumerate(markets, 1):
            for token in client.get_holders(m["conditionId"], limit=holders_per_outcome):
                for h in token["holders"]:
                    add(h["proxy_wallet"], h.get("name") or h.get("pseudonym"), source)
            if i % 250 == 0:
                print(f"    {source}: {i}/{len(markets)} markets, {len(wallets)} wallets so far")

    # Leaderboards: PNL finds top earners, VOL finds heavy traders whether they win or lose.
    for order_by in ("PNL", "VOL"):
        for period in ("ALL", "MONTH", "WEEK"):
            leaders = client.get_leaderboard(time_period=period, order_by=order_by, max_items=n_leaderboard)
            label = "leaderboard" if order_by == "PNL" else "leaderboard_vol"
            for row in leaders:
                add(row["user_id"], row.get("user_name"), f"{label}_{period.lower()}")
            print(f"  leaderboard {order_by} {period}: {len(leaders)} wallets")

    markets = client.get_markets(closed=False, order="volumeNum", ascending=False, max_items=n_markets)
    add_holders(markets, "market_holder")
    print(f"  holders of top {len(markets)} open markets collected")

    if n_closed_markets:
        closed_markets = client.get_markets(closed=True, order="volumeNum", ascending=False, max_items=n_closed_markets)
        add_holders(closed_markets, "closed_market_holder")
        print(f"  holders of top {len(closed_markets)} resolved markets collected")

    if n_recent_trades:
        trades = client.get_trades(max_items=n_recent_trades)
        for t in trades:
            add(t["proxy_wallet"], t.get("name") or t.get("pseudonym"), "recent_trader")
        print(f"  {len({t['proxy_wallet'] for t in trades})} wallets from the last {len(trades)} trades")

    return wallets


def resolved_bets(wallet, max_closed, max_open):
    """
    All of a wallet's resolved bets as one DataFrame with columns
    end_date, avg_price, total_size, pnl.

    Closed positions alone are badly biased toward wins: a winning bet gets
    redeemed for cash (and becomes "closed"), but a losing bet pays nothing, so
    many users never redeem it and it stays among the open positions as
    redeemable with current_price 0. We add those back in.
    """
    client = get_client()
    cols = ["end_date", "avg_price", "total_size", "pnl"]

    # A cap of 0 means no cap: fetch the wallet's full history
    start = time.time()
    closed = pd.DataFrame(client.get_closed_positions(wallet, max_items=max_closed or None))
    closed_capped = bool(max_closed) and len(closed) >= max_closed
    if len(closed) >= 10000:
        # Bots can have hundreds of thousands of positions; say so, since these take a while
        print(f"    big wallet {wallet}: {len(closed)} closed positions ({time.time() - start:.0f}s)")
    if not closed.empty:
        closed["pnl"] = closed["realized_pnl"]
        closed = closed[cols]

    open_ = pd.DataFrame(client.get_positions(wallet, max_items=max_open or None))
    open_capped = bool(max_open) and len(open_) >= max_open
    n_unresolved = 0
    if not open_.empty:
        n_unresolved = int((~open_["redeemable"]).sum())
        open_ = open_[open_["redeemable"]].copy()
        # total_pnl = unrealized loss/gain at resolution plus anything realized from earlier partial sells
        open_["pnl"] = open_["total_pnl"]
        open_ = open_[cols]
        # If the closed history was cut off, drop unredeemed bets older than it so both cover the same period.
        if closed_capped:
            open_ = open_[open_["end_date"] >= closed["end_date"].min()]

    # Drop tokens the wallet never bought (e.g. Polymarket's own contracts that mint shares).
    # total_size is the lifetime number of shares bought.
    closed = closed[closed["total_size"] > 0] if not closed.empty else closed
    open_ = open_[open_["total_size"] > 0] if not open_.empty else open_

    bets = pd.concat([d for d in (closed, open_) if not d.empty], ignore_index=True) if (
        not closed.empty or not open_.empty) else pd.DataFrame(columns=cols)
    return bets, len(closed), len(open_), n_unresolved, closed_capped or open_capped


def wallet_stats(wallet, max_closed, max_open=5000):  # 5000 is the maximum number of trades it will look at
    """Summarize one wallet's resolved bets."""
    df, n_closed, n_unredeemed, n_unresolved, truncated = resolved_bets(wallet, max_closed, max_open)
    base = {"wallet": wallet, "n_bets": len(df), "n_closed": n_closed,
            "n_unredeemed": n_unredeemed, "n_open_unresolved": n_unresolved,
            "history_truncated": truncated}
    if df.empty:
        return base

    cost = (df["avg_price"] * df["total_size"]).sum()
    wins = int((df["pnl"] > 0).sum())
    win_rate = wins / len(df)
    # If the market prices were fair, a bet bought at price p wins with probability p.
    # So the average entry price is the win rate you'd "expect"; beating it is the edge.
    avg_entry = df["avg_price"].mean()
    return {
        **base,
        "wins": wins,
        "losses": len(df) - wins,
        "win_rate": win_rate,
        "avg_entry_price": avg_entry,
        "win_rate_edge": win_rate - avg_entry,
        "total_pnl": df["pnl"].sum(),
        "total_cost": cost,
        "roi": df["pnl"].sum() / cost if cost else None,
        "median_pnl": df["pnl"].median(),
        "first_end_date": df["end_date"].min(),
        "last_end_date": df["end_date"].max(),
    }


def skipped_path(out_path):
    """Side file listing wallets that were below --min-bets, so later runs don't check them again."""
    return os.path.splitext(out_path)[0] + ".skipped.txt"


def checked_stats(wallet, max_closed, max_open, min_bets):
    """
    wallet_stats, or None if the wallet has fewer than min_bets resolved bets
    or is one of Polymarket's own contracts (which the v2 API refuses to look up).
    """
    try:
        # Cheap first check: one request for the number of markets traded. That count also
        # includes open markets, so it's usually at least n_bets and rarely skips a wallet that would pass.
        if min_bets and get_client().get_traded(wallet) < min_bets:
            return None
        row = wallet_stats(wallet, max_closed, max_open)
    except requests.HTTPError as e:
        if e.response is not None and "protocol contract" in e.response.text:
            return None
        raise
    return row if row["n_bets"] >= min_bets else None


def fetch_wallets(todo, info, args, out_path, min_bets=0):
    """
    Fetch stats for each wallet in todo and append them to out_path in batches.
    info maps wallet -> {"name": ..., "sources": "a;b"}. Wallets with fewer than
    min_bets resolved bets are not saved; they're listed in skipped_path(out_path) instead.
    Returns the number that failed.
    """
    start = time.time()
    batch = []
    skipped = []
    n_skipped = 0
    failed = 0

    def flush():
        if batch:
            pd.DataFrame(batch).reindex(columns=COLUMNS).to_csv(out_path, mode="a", header=not has_rows(out_path), index=False)
            batch.clear()
        if skipped:
            with open(skipped_path(out_path), "a") as f:
                f.writelines(w + "\n" for w in skipped)
            skipped.clear()

    # Shuffle so the heavy bot wallets (which come first, from the volume leaderboard)
    # are spread out instead of tying up every worker at the start
    todo = list(todo)
    random.shuffle(todo)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(checked_stats, w, args.max_closed, args.max_open, min_bets): w for w in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            w = futures[fut]
            try:
                row = fut.result()
            except Exception as e:
                failed += 1
                print(f"  failed {w}: {e}")
                continue
            if row is None:
                skipped.append(w)
                n_skipped += 1
            else:
                row["name"] = info[w]["name"]
                row["sources"] = info[w]["sources"]
                batch.append(row)
            if len(batch) + len(skipped) >= 100:
                flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} wallets, {n_skipped} skipped under {min_bets} bets  ({time.time() - start:.0f}s)")
    flush()
    return failed


def refetch_truncated(args):
    """
    Refetch wallets whose history hit a cap, using the current --max-closed/--max-open,
    and replace their rows in args.out. Their name and sources are kept.

    Progress goes to a side file (<out>.refetch.csv) first, so an interrupted run
    can be rerun and picks up where it left off. The main CSV is only rewritten at the end.
    """
    if not has_rows(args.out):
        print(f"{args.out} is missing or empty; nothing to refetch.")
        return
    existing = pd.read_csv(args.out).rename(columns=OLD_NAMES)
    truncated =existing["history_truncated"].astype(str) == "True"
    # Only wallets the new cap would actually fetch more of (0 = no cap)
    more_room = (args.max_closed == 0) | (existing["n_closed"] < args.max_closed)
    targets = existing[truncated & more_room]
    print(f"{truncated.sum()} truncated wallets in {args.out}, {len(targets)} to refetch "
          f"with --max-closed {args.max_closed or 'no cap'} --max-open {args.max_open or 'no cap'}")

    progress_path = os.path.splitext(args.out)[0] + ".refetch.csv"
    done = set()
    if has_rows(progress_path):
        done = set(pd.read_csv(progress_path, usecols=["wallet"])["wallet"])
        print(f"Resuming: {len(done)} already refetched in {progress_path}")

    info = {w: {"name": n, "sources": s}
            for w, n, s in zip(targets["wallet"], targets["name"], targets["sources"])}
    todo = [w for w in info if w not in done]
    failed = fetch_wallets(todo, info, args, progress_path) if todo else 0

    if not has_rows(progress_path):
        print("Nothing was refetched.")
        return
    new = pd.read_csv(progress_path)
    new = new[new["wallet"].isin(info)]
    merged = pd.concat([existing[~existing["wallet"].isin(new["wallet"])], new], ignore_index=True)
    # Write to a temp file and swap it in, so a crash mid-write can't corrupt the main CSV
    tmp_path = args.out + ".tmp"
    merged.reindex(columns=COLUMNS).to_csv(tmp_path, index=False)
    os.replace(tmp_path, args.out)
    os.remove(progress_path)
    print(f"\nReplaced {len(new)} rows in {args.out} ({failed} failed; rerun to retry them)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Defaults are set to the API's own limits (checked Oct 2026), to find as many wallets as possible:
    # leaderboard stops at rank 10,000; market lists stop around 2,000 deep; holders max out
    # around 500 per outcome; recent trades stop at 10,000.    #10000; 2000; 500; 2000; 10000...
    parser.add_argument("--leaderboard", type=int, default=10000, help="top N users from each leaderboard period (API max 10,000)")
    parser.add_argument("--markets", type=int, default=2000, help="number of top open markets to pull holders from (API max ~2,000)")
    parser.add_argument("--holders", type=int, default=500, help="top holders per outcome of each market (API max ~500)")
    parser.add_argument("--closed-markets", type=int, default=200, help="number of top resolved markets to pull holders from (0 to skip; API max ~2,000)")
    parser.add_argument("--recent-trades", type=int, default=5000, help="number of most recent trades to pull traders from (0 to skip; API max 10,000)")
    parser.add_argument("--max-closed", type=int, default=5000, help="most recent closed positions per wallet (0 = no cap)")
    parser.add_argument("--max-open", type=int, default=5000, help="open/unredeemed positions per wallet (0 = no cap)")
    parser.add_argument("--workers", type=int, default=6, help="parallel requests (keep low to avoid rate limits)")
    parser.add_argument("--min-bets", type=int, default=50, help="only save wallets with at least this many resolved bets (0 = save all)")
    parser.add_argument("--out", default="data/wallets_50+_raw.csv")
    parser.add_argument("--fresh", action="store_true", help="ignore an existing output file and start over")
    parser.add_argument("--refetch-truncated", action="store_true", help="don't collect new wallets; refetch the capped ones in --out with the current caps")
    args = parser.parse_args()

    if args.refetch_truncated:
        refetch_truncated(args)
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if has_rows(args.out) and not args.fresh:
        done = set(pd.read_csv(args.out, usecols=["wallet"])["wallet"])
        print(f"Resuming: {len(done)} wallets already in {args.out}")
    elif os.path.exists(args.out):
        os.remove(args.out)  # --fresh, or an empty file left behind by a crash
    if args.fresh and os.path.exists(skipped_path(args.out)):
        os.remove(skipped_path(args.out))
    # Wallets already found to be under --min-bets. Delete this file after lowering --min-bets.
    if os.path.exists(skipped_path(args.out)):
        with open(skipped_path(args.out)) as f:
            below = {line.strip() for line in f if line.strip()}
        done |= below
        print(f"Skipping {len(below)} wallets already found under the bet cutoff ({skipped_path(args.out)})")

    print("Collecting wallets...")
    wallets = collect_wallets(get_client(), args.leaderboard, args.markets, args.holders,
                              args.closed_markets, args.recent_trades)
    todo = [w for w in wallets if w not in done]
    print(f"{len(wallets)} unique wallets, {len(todo)} to fetch\n")

    info = {w: {"name": v["name"], "sources": ";".join(sorted(v["sources"]))} for w, v in wallets.items()}
    failed = fetch_wallets(todo, info, args, args.out, min_bets=args.min_bets)

    if not has_rows(args.out):
        print(f"\nNothing saved ({failed} failed)")
        return
    df = pd.read_csv(args.out)
    print(f"\nSaved {len(df)} wallets to {args.out} ({failed} failed)")
    solid = df[df["n_bets"] >= 30].sort_values("win_rate_edge", ascending=False)
    print(f"\nTop 10 by win-rate edge (wallets with 30+ resolved bets, {len(solid)} total):")
    print(solid[["wallet", "name", "n_bets", "n_unredeemed", "win_rate", "avg_entry_price", "win_rate_edge", "total_pnl"]]
          .head(10).to_string(index=False))


if __name__ == "__main__":
    main()
