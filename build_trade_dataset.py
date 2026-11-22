"""
Build a trade-by-trade CSV: one row per action (each buy, sell, split, merge,
redemption and conversion) for a set of wallets, with facts about each market
and its final result.

By default it takes 500 random wallets from data/aggregate_data/wallets_analysis1.csv. Raising
--sample later keeps those 500 and adds more, so a rerun extends the same file.
Each wallet's most recent --max-actions actions are kept (newest first); wallets
that hit the cap are marked history_truncated.

Run with:
    .venv\\Scripts\\python build_trade_dataset.py
    .venv\\Scripts\\python build_trade_dataset.py --sample 2000
    .venv\\Scripts\\python build_trade_dataset.py --wallets-from data/bets_sample.csv --sample 0 --out data/trades_sample.csv

It works in two steps:
  1. Download each wallet's actions (and its positions' fees) into --out.
     Reruns skip wallets already there; --fresh starts the file over.
  2. Look up every market in the file and add the market columns.
     Resolved markets are saved in --market-cache (data/market_info.csv) and reused.
     The finished data is saved as Parquet (--out with .parquet, e.g. data/trades_raw.parquet);
     the CSV stays as the raw download. --output csv or both also/instead rewrites the CSV.

If some market lookups failed, fill in just those afterwards (fast; reads the Parquet file):
    .venv\\Scripts\\python build_trade_dataset.py --fill-missing

Reading the Parquet file (much faster than the CSV, and it can load just what you need):
    trades = pd.read_parquet("data/trades_raw.parquet", columns=["wallet", "price", "won"])
    one = pd.read_parquet("data/trades_raw.parquet", filters=[("wallet", "==", "0x...")])

Columns:
  wallet, name          the wallet and its username
  type                  TRADE   buy or sell shares with another user (side = BUY or SELL)
                        SPLIT   put in $1 per pair, get 1 Yes + 1 No share
                        MERGE   hand back 1 Yes + 1 No share per pair, get $1
                        REDEEM  after the market resolved, cash shares in ($1 per winning share)
                        CONVERSION  in a group of mutually exclusive markets, swap No shares
                                for Yes shares in the other markets of the group
  side                  BUY or SELL for trades, blank otherwise
  time                  when it happened (UTC)
  title, slug           the market; event_slug groups related markets
  outcome, outcome_index  the outcome traded (e.g. "Yes", 0). Blank / 999 for actions
                        that touch both outcomes (SPLIT, MERGE, CONVERSION)
  condition_id          the market; token_id is the outcome traded
  price                 price per share (trades; 0.5 for splits; 0 where no price applies)
  size                  number of shares
  usdc_size             money that changed hands, in USDC (price x size for trades)
  transaction_hash      the on-chain transaction
  is_combo              True for trades that were part of a combined (multi-market) order
  history_truncated     True if this wallet hit --max-actions, so older actions are missing

Position columns (the wallet's WHOLE position in this outcome, repeated on each of its
rows; don't add them up across rows):
  position_entry_fees_usdc   trading fees paid on all purchases of this outcome
  position_total_cost_usdc   money spent on all purchases of this outcome, fees included

Market columns (facts about the market, the same on every row of that market):
  event_id              the event (page) the market belongs to; related markets share it
  negative_risk         True if the market is in a group where only one market can resolve Yes
  end_date              the market's scheduled end date
  opposite_token_id     the other outcome of the market (blank for SPLIT/MERGE/CONVERSION)
  current_price         this outcome's latest price; its final payout once resolved
  market_resolved       True if the market has resolved
  resolved_time         when it was settled (UTC)
  payout_per_share      what one share of this outcome paid: 1 (won), 0 (lost), or
                        something in between for markets settled 50/50. Blank if unresolved
  won                   True if this outcome won, False if it lost, blank otherwise
  profit_if_held        BUY rows only: (payout_per_share - price) x size, the profit this
                        purchase would make if held to the end. Excludes trading fees
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import pyarrow

from build_wallet_dataset import get_client, has_rows

ACTION_TYPES = ["TRADE", "SPLIT", "MERGE", "REDEEM", "CONVERSION"]
ACTION_COLUMNS = [
    "wallet", "name", "type", "side", "time", "title", "slug", "event_slug", "outcome", "outcome_index",
    "condition_id", "token_id", "price", "size", "usdc_size", "transaction_hash", "is_combo", "history_truncated",
    "position_entry_fees_usdc", "position_total_cost_usdc",
]
MARKET_COLUMNS = [
    "event_id", "negative_risk", "end_date", "opposite_token_id", "current_price",
    "market_resolved", "resolved_time", "payout_per_share", "won", "profit_if_held",
]


def wallet_actions(wallet, max_actions):
    """One wallet's most recent actions as a DataFrame with ACTION_COLUMNS (except name)."""
    client = get_client()
    raw = pd.DataFrame(client.get_activity(wallet, types=ACTION_TYPES, max_items=max_actions or None))
    if raw.empty:
        return raw
    df = raw.reindex(columns=ACTION_COLUMNS)
    df["wallet"] = wallet
    df["time"] = pd.to_datetime(raw["timestamp"], unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["is_combo"] = df["is_combo"].eq(True)  # blank on ordinary trades
    df["side"] = df["side"].replace("", None)
    df["history_truncated"] = bool(max_actions) and len(df) >= max_actions

    # Fees are only reported per position, so download the wallet's positions and attach
    # each position's totals to its rows
    positions = pd.DataFrame(client.get_closed_positions(wallet, max_items=max_actions or None)
                             + client.get_positions(wallet, max_items=max_actions or None, sort_by="TIMESTAMP"))
    if not positions.empty:
        # A position can come back in both lists; keep one copy
        fees = positions.drop_duplicates("token_id").set_index("token_id")
        df["position_entry_fees_usdc"] = df["token_id"].map(fees["entry_fees_usdc"])
        df["position_total_cost_usdc"] = df["token_id"].map(fees["total_cost_usdc"])
    return df


def download_actions(wallets, args):
    """Step 1: append each wallet's actions to args.out. Returns the number of wallets that failed."""
    done = set()
    if has_rows(args.out):
        for chunk in read_chunks(args.out, usecols=["wallet"]):
            done.update(chunk["wallet"].unique())
        print(f"Resuming: {len(done)} wallets already in {args.out}")
    todo = [w for w in wallets if w not in done]
    print(f"{len(wallets)} wallets, {len(todo)} to download\n")

    start = time.time()
    n_rows = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(wallet_actions, w, args.max_actions): w for w in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            w = futures[fut]
            try:
                actions = fut.result()
            except requests.HTTPError as e:
                if e.response is not None and "protocol contract" in e.response.text:
                    continue  # one of Polymarket's own contracts; the API won't look it up
                failed += 1
                print(f"  failed {w}: {e}")
                continue
            except Exception as e:
                failed += 1
                print(f"  failed {w}: {e}")
                continue
            if not actions.empty:
                actions["name"] = wallets[w]
                # Written one wallet at a time, so an interrupted run only loses the wallet in progress
                actions[ACTION_COLUMNS].to_csv(args.out, mode="a", header=not has_rows(args.out), index=False)
                n_rows += len(actions)
                if actions["history_truncated"].iloc[0]:
                    print(f"    {w} hit the cap of {args.max_actions} actions")
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} wallets, {n_rows} actions  ({time.time() - start:.0f}s)")
    return failed


def market_info(condition_ids, workers, cache_path):
    """
    Facts about each market as a DataFrame with condition_id, event_id, negative_risk, end_date,
    token_0/token_1, price_0/price_1, market_resolved and resolved_time.
    Resolved markets are cached in cache_path; open ones are looked up again every run,
    since their prices change.
    """
    cached = pd.read_csv(cache_path, dtype={"token_0": str, "token_1": str, "event_id": str}) \
        if has_rows(cache_path) else pd.DataFrame(columns=["condition_id"])
    todo = set(condition_ids) - set(cached["condition_id"])
    # A few conversion rows have ids written as "\x..." (a database format) instead of "0x...".
    # They aren't real market ids, and their backslashes make the lookup URL too long (414 error)
    malformed = {c for c in todo if not str(c).startswith("0x")}
    todo = sorted(todo - malformed)
    print(f"\nMarket info: {len(set(condition_ids)) - len(todo) - len(malformed)} cached, {len(todo)} to look up"
          + (f", {len(malformed)} malformed ids skipped (left blank)" if malformed else ""))

    def lookup(batch):
        # Gamma lists closed and open markets separately, so ask for each
        client = get_client()
        markets = client.get_markets(closed=True, max_items=len(batch), condition_ids=batch)
        missing = set(batch) - {m["conditionId"] for m in markets}
        if missing:
            markets += client.get_markets(closed=False, max_items=len(missing), condition_ids=sorted(missing))
        out = []
        for m in markets:
            prices = [float(p) for p in m.get("outcomePrices") or []]
            tokens = m.get("clobTokenIds") or []
            if len(prices) != 2 or len(tokens) != 2:
                continue
            resolved = m.get("closed", False) and (
                m.get("umaResolutionStatus") == "resolved" or sorted(prices) in ([0.0, 1.0], [0.5, 0.5]))
            events = m.get("events") or [{}]
            out.append({
                "condition_id": m["conditionId"], "event_id": events[0].get("id"),
                "negative_risk": bool(m.get("negRisk")),
                "end_date": m.get("endDateIso") or (m.get("endDate") or "")[:10] or None,
                "token_0": tokens[0], "token_1": tokens[1], "price_0": prices[0], "price_1": prices[1],
                "market_resolved": resolved, "resolved_time": m.get("closedTime") if resolved else None,
            })
        return out

    batches = [todo[i:i + 100] for i in range(0, len(todo), 100)]  # Gamma accepts about 100 ids per request
    found = []
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(lookup, b) for b in batches]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rows = fut.result()
            except Exception as e:
                # Skip the batch instead of stopping; its markets are looked up again on the next run
                failed += 1
                print(f"  batch failed ({failed} so far): {e}")
                continue
            found.extend(rows)
            # Save resolved markets as they come in, so an interrupted run doesn't lose them
            resolved = [r for r in rows if r["market_resolved"]]
            if resolved:
                pd.DataFrame(resolved).to_csv(cache_path, mode="a", header=not has_rows(cache_path), index=False)
            if i % 50 == 0 or i == len(batches):
                print(f"  {i}/{len(batches)} batches, {len(found)} markets found")
    if failed:
        print(f"  {failed} batches failed; their markets are left blank. Rerun to fill them in "
              f"(resolved markets found so far are saved in {cache_path}).")

    return pd.concat([cached, pd.DataFrame(found)], ignore_index=True).drop_duplicates("condition_id")


CHUNK_ROWS = 1_000_000  # rows read into memory at a time; the full file can be far bigger than RAM


def read_chunks(path, **kwargs):
    """Read a CSV CHUNK_ROWS rows at a time, keeping the long ids as text."""
    return pd.read_csv(path, chunksize=CHUNK_ROWS, float_precision="round_trip",
                       dtype={"token_id": str, "opposite_token_id": str, "event_id": str}, **kwargs)


# Fixed type of every column in the Parquet file. Each chunk is converted to these types,
# so all chunks line up even when a chunk happens to have a column that's entirely blank.
COLUMN_TYPES = {
    **{c: "string" for c in ["wallet", "name", "type", "side", "time", "title", "slug", "event_slug",
                             "outcome", "condition_id", "token_id", "transaction_hash", "event_id",
                             "end_date", "opposite_token_id", "resolved_time"]},
    **{c: "float64" for c in ["price", "size", "usdc_size", "position_entry_fees_usdc",
                              "position_total_cost_usdc", "current_price", "payout_per_share", "profit_if_held"]},
    **{c: "boolean" for c in ["is_combo", "history_truncated", "negative_risk", "market_resolved", "won"]},
    "outcome_index": "Int64",
}


def add_market_columns(args):
    """
    Step 2: add the market columns to every row of args.out and save the result as
    Parquet (args.parquet_out), CSV (rewriting args.out) or both, depending on args.output.
    The file is processed CHUNK_ROWS rows at a time, so it works however large the file is.
    """
    # First pass: every market in the file (only the condition_id column is held in memory)
    condition_ids = set()
    for chunk in read_chunks(args.out, usecols=["condition_id"]):
        condition_ids.update(chunk["condition_id"].dropna().unique())
    markets = market_info(condition_ids, args.workers, args.market_cache)

    # Second pass: add the columns chunk by chunk, writing to temp files that are swapped in
    # at the end, so a crash mid-write can't corrupt an existing file
    write_csv = args.output in ("csv", "both")
    write_parquet = args.output in ("parquet", "both")
    targets = ([args.out] if write_csv else []) + ([args.parquet_out] if write_parquet else [])
    print(f"\nAdding market columns, saving to {' and '.join(targets)}...")
    csv_tmp, parquet_tmp = args.out + ".tmp", args.parquet_out + ".tmp"
    parquet_writer = None
    n_rows = n_resolved = n_info = 0
    for i, chunk in enumerate(read_chunks(args.out), 1):
        df = with_market_columns(chunk, markets)
        if write_csv:
            df.to_csv(csv_tmp, mode="w" if i == 1 else "a", header=i == 1, index=False)
        if write_parquet:
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = pa.Table.from_pandas(df.astype(COLUMN_TYPES), preserve_index=False)
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(parquet_tmp, table.schema, compression="zstd")
            parquet_writer.write_table(table)  # each chunk becomes one row group
        n_rows += len(df)
        n_resolved += int(df["market_resolved"].sum())
        n_info += int(df["event_id"].notna().sum())
        if i % 10 == 0:
            print(f"  {n_rows:,} rows done")
    if parquet_writer is not None:
        parquet_writer.close()
        os.replace(parquet_tmp, args.parquet_out)
    if write_csv:
        os.replace(csv_tmp, args.out)
    print(f"\nSaved {n_rows:,} actions to {' and '.join(targets)}: "
          f"{n_resolved / n_rows:.0%} in resolved markets, {n_info / n_rows:.0%} with market info")


def fill_missing(args):
    """
    --fill-missing: look up only the markets whose rows have blank market columns (e.g. from
    batches that failed), and rewrite the Parquet file with them filled in. Reads the Parquet
    file instead of the CSV, and copies row groups with nothing to fill unchanged, so it's
    much faster than a full rerun. Other rows, including open markets' prices, are not updated.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not os.path.exists(args.parquet_out):
        print(f"{args.parquet_out} doesn't exist yet; run without --fill-missing first.")
        return
    source = pq.ParquetFile(args.parquet_out)

    # First pass: markets with a blank event_id, i.e. no market info (only two columns are read)
    blank = set()
    for i in range(source.num_row_groups):
        ids = source.read_row_group(i, columns=["condition_id", "event_id"]).to_pandas()
        blank.update(ids.loc[ids["event_id"].isna(), "condition_id"].dropna().unique())
    print(f"{len(blank)} markets in {args.parquet_out} have no market info")
    if not blank:
        return
    markets = market_info(blank, args.workers, args.market_cache)
    markets = markets[markets["condition_id"].isin(blank) & markets["event_id"].notna()]
    if markets.empty:
        print("None of them were found, so the file is left as it is.")
        return
    filled = set(markets["condition_id"])

    # Second pass: rewrite to a temp file, recomputing the market columns only for the blank rows
    print(f"\nFilling in {len(filled)} markets, rewriting {args.parquet_out}...")
    tmp = args.parquet_out + ".tmp"
    n_fixed = 0
    with pq.ParquetWriter(tmp, source.schema_arrow, compression="zstd") as writer:
        for i in range(source.num_row_groups):
            table = source.read_row_group(i)
            ids = table.select(["condition_id", "event_id"]).to_pandas()
            mask = ids["event_id"].isna() & ids["condition_id"].isin(filled)
            if mask.any():
                df = table.to_pandas()
                fixed = with_market_columns(df[mask], markets)
                fixed.index = df.index[mask]
                df = df.astype(COLUMN_TYPES)
                df.loc[mask, MARKET_COLUMNS] = fixed[MARKET_COLUMNS].astype(
                    {c: COLUMN_TYPES[c] for c in MARKET_COLUMNS})
                table = pa.Table.from_pandas(df.astype(COLUMN_TYPES), schema=source.schema_arrow,
                                             preserve_index=False)
                n_fixed += int(mask.sum())
            writer.write_table(table)  # row groups with nothing to fill are copied as they are
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{source.num_row_groups} row groups done")
    source.close()
    os.replace(tmp, args.parquet_out)
    print(f"\nFilled in market info for {n_fixed:,} rows in {args.parquet_out}")


def with_market_columns(actions, markets):
    """One chunk of actions with MARKET_COLUMNS (re)computed from the market info table."""
    actions = actions.drop(columns=[c for c in MARKET_COLUMNS if c in actions.columns])
    df = actions.merge(markets, on="condition_id", how="left")
    df["market_resolved"] = df["market_resolved"].eq(True)  # markets not found are treated as unresolved
    df["resolved_time"] = pd.to_datetime(df["resolved_time"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    # Columns for the outcome this row traded (outcome_index 0 or 1; 999 = both outcomes)
    first, second = df["outcome_index"] == 0, df["outcome_index"] == 1
    df["opposite_token_id"] = df["token_1"].where(first, df["token_0"].where(second))
    df["current_price"] = df["price_0"].where(first, df["price_1"].where(second))
    df["payout_per_share"] = df["current_price"].where(df["market_resolved"])
    df["won"] = df["payout_per_share"].map({1.0: True, 0.0: False})
    is_buy = (df["type"] == "TRADE") & (df["side"] == "BUY")
    df["profit_if_held"] = ((df["payout_per_share"] - df["price"]) * df["size"]).where(is_buy)
    return df[ACTION_COLUMNS + MARKET_COLUMNS]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wallets-from", default="data/aggregate_data/wallets_analysis1.csv", help="CSV whose wallet (and name) columns give the wallets")
    parser.add_argument("--sample", type=int, default=0, help="number of random wallets from that file (0 = all of them)")
    parser.add_argument("--seed", type=int, default=42, help="random seed, so reruns pick the same wallets")
    parser.add_argument("--max-actions", type=int, default=20000, help="most recent actions per wallet (0 = no cap)")
    parser.add_argument("--workers", type=int, default=4, help="parallel requests (keep low to avoid rate limits)")
    parser.add_argument("--out", default="data/trades_raw.csv")
    parser.add_argument("--market-cache", default="data/market_info.csv", help="where resolved markets' info is saved for reuse")
    parser.add_argument("--fresh", action="store_true", help="delete --out and download every wallet again")
    parser.add_argument("--output", choices=["parquet", "csv", "both"], default="parquet",
                        help="where the finished data (with market columns) goes: a Parquet file, "
                             "the CSV itself, or both. With parquet, the CSV is left as the raw download")
    parser.add_argument("--parquet-out", help="Parquet file to write (default: --out with .parquet)")
    parser.add_argument("--fill-missing", action="store_true",
                        help="only look up the markets left blank in the Parquet file (e.g. from failed "
                             "batches) and fill them in; skips downloading and doesn't read the CSV")
    args = parser.parse_args()
    args.parquet_out = args.parquet_out or os.path.splitext(args.out)[0] + ".parquet"
    if args.output != "csv":
        try:
            import pyarrow  # noqa: F401  (needed for Parquet; checked now so a long run doesn't fail at the end)
        except ImportError:
            print("Saving Parquet needs the pyarrow package. Install it with:\n"
                  "    .\\.venv\\Scripts\\python -m pip install pyarrow\n"
                  "or run with --output csv.")
            return
    if args.fill_missing:
        fill_missing(args)
        return

    if args.fresh and os.path.exists(args.out):
        os.remove(args.out)
    elif has_rows(args.out):
        # A file made by an older version of this script is missing columns that resuming can't fill in
        missing = [c for c in ACTION_COLUMNS if c not in pd.read_csv(args.out, nrows=0).columns]
        if missing:
            print(f"{args.out} was made by an older version of this script and has no {', '.join(missing)} columns.\n"
                  f"Rerun with --fresh to download it again with them.")
            return

    source = pd.read_csv(args.wallets_from, usecols=["wallet", "name"]).drop_duplicates("wallet")
    if args.sample:
        # Shuffle every wallet in a fixed order and take the first --sample. A larger --sample
        # later keeps the same first wallets and only adds new ones, so a rerun extends the file.
        source = source.sample(frac=1, random_state=args.seed).head(args.sample)
    wallets = dict(zip(source["wallet"], source["name"]))
    print(f"{len(wallets)} wallets from {args.wallets_from}")

    failed = download_actions(wallets, args)
    if not has_rows(args.out):
        print(f"\nNothing saved ({failed} failed)")
        return
    add_market_columns(args)
    if failed:
        print(f"{failed} wallets failed; rerun to retry them")


if __name__ == "__main__":
    main()
