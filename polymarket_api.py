"""
Read-only client for Polymarket's public APIs. No API key or wallet needed.

Polymarket exposes three public HTTP APIs:
  - Gamma API (gamma-api.polymarket.com): market and event metadata
      (questions, outcomes, prices, volume, open/closed status).
  - Data API  (data-api.polymarket.com/v2): user-level data. Trades, positions,
      top holders of a market, and the leaderboard. Users show up as
      `proxy_wallet` addresses (their anonymous ID).
  - CLOB API  (clob.polymarket.com):      live order book and prices.

The Data API moved to v2 on November 2, 2026 (v1 now answers 410 Gone). v2
differs from v1 in three ways that matter here: every response is wrapped as
{"data": ..., "pagination": ...}, fields are snake_case (proxy_wallet, not
proxyWallet), and pages are walked with a cursor instead of an offset. The
Gamma API is unchanged and still uses camelCase and offsets.

Key IDs:
  - conditionId / condition_id: identifies a market (one yes/no question).
  - token_id: identifies one outcome of that market (e.g. "Yes").
  - proxy_wallet: identifies a user.
"""

import json
import time

import pandas as pd
import requests

GAMMA_URL = "https://gamma-api.polymarket.com"
DATA_URL = "https://data-api.polymarket.com/v2"
CLOB_URL = "https://clob.polymarket.com"


class PolymarketClient:
    def __init__(self, timeout=20, max_retries=5, pause=0.1):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "DataScienceClub-Polymarket/0.1"
        self.timeout = timeout
        self.max_retries = max_retries
        self.pause = pause  # seconds to wait between paginated requests

    # ---------- low-level helpers ----------

    def _get(self, url, params=None):
        """GET a URL and return parsed JSON, retrying on rate limits and server errors."""
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()

    def _paginate_cursor(self, url, params, page_size=1000, max_items=None):
        """
        Walk Data API v2 pages: each response carries pagination.next_cursor,
        which is passed back to get the next page, until it is null or max_items is reached.
        """
        results = []
        cursor = None
        while max_items is None or len(results) < max_items:
            page_params = {**params, "limit": page_size}
            if cursor:
                page_params["cursor"] = cursor
            page = self._get(url, page_params)
            results.extend(page.get("data") or [])
            cursor = (page.get("pagination") or {}).get("next_cursor")
            if not cursor:
                break
            time.sleep(self.pause)
        return results[:max_items] if max_items else results

    def _paginate(self, url, params, page_size, max_items=None):
        """Walk limit/offset pages (Gamma API) until a short page comes back or max_items is reached."""
        results = []
        offset = 0
        while max_items is None or len(results) < max_items:
            page = self._get(url, {**params, "limit": page_size, "offset": offset})
            if not page:
                break
            results.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
            time.sleep(self.pause)
        return results[:max_items] if max_items else results

    # ---------- Gamma API: markets ----------

    def get_markets(self, closed=None, max_items=500, **filters):
        """
        List markets. closed=False for open markets, True for resolved ones.
        Extra filters are passed straight through, e.g. order="volumeNum"
        (not "volume", which sorts the number as text),
        ascending=False, slug="...", tag_id=..., end_date_min="2026-01-01".
        """
        params = dict(filters)
        if closed is not None:
            params["closed"] = str(closed).lower()
        markets = self._paginate(f"{GAMMA_URL}/markets", params, page_size=100, max_items=max_items)
        # outcomes, outcomePrices and clobTokenIds arrive as JSON strings; decode them
        for m in markets:
            for key in ("outcomes", "outcomePrices", "clobTokenIds"):
                if isinstance(m.get(key), str):
                    m[key] = json.loads(m[key])
        return markets

    def get_market(self, slug):
        """Fetch one market by its URL slug (the last part of a polymarket.com/event/... link)."""
        markets = self.get_markets(slug=slug, max_items=1)
        return markets[0] if markets else None

    def get_events(self, closed=None, max_items=100, **filters):
        """List events (groups of related markets, e.g. 'World Cup winner')."""
        params = dict(filters)
        if closed is not None:
            params["closed"] = str(closed).lower()
        return self._paginate(f"{GAMMA_URL}/events", params, page_size=100, max_items=max_items)

    # ---------- Data API: users and trades ----------

    def get_leaderboard(self, time_period="ALL", order_by="PNL", max_items=100, category=None):
        """
        Top users by profit or volume. Each row's wallet is in `user_id`.
        time_period: DAY, WEEK, MONTH, ALL. order_by: PNL or VOL (VOLUME also works).
        """
        params = {"time_period": time_period.lower(),
                  "sort_by": "VOLUME" if order_by.upper() in ("VOL", "VOLUME") else "PNL"}
        if category:
            params["category"] = category
        return self._paginate_cursor(f"{DATA_URL}/leaderboard", params, max_items=max_items)

    def get_trades(self, market=None, user=None, side=None, max_items=1000):
        """
        Recent trades, newest first. Filter by market (conditionId), user
        (proxy_wallet) and/or side ("BUY" or "SELL").
        """
        params = {}
        if market:
            params["condition"] = market
        if user:
            params["user"] = user
        if side:
            params["side"] = side
        return self._paginate_cursor(f"{DATA_URL}/trades", params, max_items=max_items)

    def get_positions(self, user, max_items=500, status="OPEN", sort_by=None):
        """
        A user's positions. status:
          OPEN     every position still held, including ones in resolved markets
                   that were never cashed in (those have redeemable=True)
          CLOSED   positions the user fully exited (sold, or redeemed after resolution)
          REDEEMABLE / REDEEMABLE_LOST   only the resolved-but-not-cashed-in ones (all / losers)
        """
        params = {"user": user, "status": status}
        if sort_by:
            params.update(sort_by=sort_by, sort_direction="DESC")
        return self._paginate_cursor(f"{DATA_URL}/positions", params, max_items=max_items)

    def get_closed_positions(self, user, max_items=1000, sort_by="TIMESTAMP"):
        """
        A user's closed positions, newest first, each with realized_pnl. This is
        the history used for win rates. Sort by timestamp (not the API default,
        REALIZED_PNL) so a truncated history is a fair sample, not just the biggest wins.
        """
        return self.get_positions(user, max_items=max_items, status="CLOSED", sort_by=sort_by)

    def get_traded(self, user):
        """Number of markets a user has ever traded (open and resolved), in one request."""
        stats = self._get(f"{DATA_URL}/user-stats", {"user": user})["data"]
        return (stats or {}).get("trades") or 0  # data is null for unknown users

    def get_holders(self, market, limit=20):
        """Top holders (up to 1000) of each outcome token in a market (conditionId)."""
        return self._get(f"{DATA_URL}/holders", {"condition": market, "limit": limit})["data"]

    # ---------- CLOB API: live prices ----------

    def get_midpoint(self, token_id):
        """Current midpoint price (0 to 1) of one outcome token."""
        return float(self._get(f"{CLOB_URL}/midpoint", {"token_id": token_id})["mid"])

    def get_order_book(self, token_id):
        """Full order book (bids and asks) for one outcome token."""
        return self._get(f"{CLOB_URL}/book", {"token_id": token_id})

    # ---------- analysis helpers ----------

    def user_win_rate(self, user, max_items=1000):
        """
        Summarize a user's most recent closed positions (up to max_items).
        A position counts as a win when its realized_pnl is positive.
        """
        closed = self.get_closed_positions(user, max_items=max_items)
        if not closed:
            return {"user": user, "n_positions": 0, "wins": 0, "win_rate": None, "total_pnl": 0.0}
        pnl = [p["realized_pnl"] for p in closed]
        wins = sum(1 for x in pnl if x > 0)
        return {
            "user": user,
            "n_positions": len(closed),
            "wins": wins,
            "win_rate": wins / len(closed),
            "total_pnl": sum(pnl),
        }


def to_df(records):
    """Turn a list of API records into a pandas DataFrame."""
    return pd.DataFrame(records)
