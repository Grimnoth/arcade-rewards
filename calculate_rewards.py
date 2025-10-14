import argparse
import csv
import datetime as dt
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, getcontext
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "weekly-input"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT

# Increase Decimal precision for ETH math (beyond 18 to avoid rounding accumulation)
getcontext().prec = 50


@dataclass(frozen=True)
class EthBand:
    leaderboard: str  # "total" or "best"
    start_rank: int
    end_rank: int
    winners: int
    share_fraction: Decimal  # 0..1 as Decimal


@dataclass
class LeaderboardData:
    # canonical lowercase wallet for joins -> original case wallet from source
    wallet_lower_to_original: Dict[str, str]
    # scores per wallet (sum for total, best single for best_run)
    value_by_wallet_lower: Dict[str, Decimal]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arcade Rewards CSV processor")
    parser.add_argument("--season", required=True, help="Season tag, e.g. s10")
    parser.add_argument("--eth-total", type=Decimal, required=True,
                        help="Total ETH to distribute across leaderboards")
    parser.add_argument("--eth-split", default="85:15",
                        help="Split between total:best in percent format, e.g. 85:15")
    parser.add_argument("--min-credits", type=int, default=20,
                        help="Minimum credits spent to qualify (default 20)")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Weekly input directory")
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR), help="Config directory")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--date", default=None, help="YYYYMMDD override for output filenames")
    parser.add_argument("--dry-run", action="store_true", help="Compute but do not write files")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warn"], help="Log verbosity")
    parser.add_argument("--eth-decimals", type=int, default=18, help="Decimal places for ETH output (default 18)")
    parser.add_argument("--eth-target-winners", type=int, default=None, help="Optional target number of ETH winners (post-merge)")
    return parser.parse_args()


def log(level: str, message: str, desired: str) -> None:
    order = {"debug": 10, "info": 20, "warn": 30}
    if order[level] >= order[desired]:
        print(message)


def to_decimal(value: object) -> Optional[Decimal]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return None
    # Remove commas
    s = s.replace(",", "")
    try:
        return Decimal(s)
    except Exception:
        return None


def normalize_wallet(addr: str) -> Optional[str]:
    if addr is None:
        return None
    s = str(addr).strip()
    if s == "":
        return None
    return s.lower()


def read_blacklist(input_dir: pathlib.Path) -> List[str]:
    wallets: List[str] = []
    for txt in input_dir.glob("*.txt"):
        try:
            content = txt.read_text(encoding="utf-8")
        except Exception:
            content = txt.read_text(errors="ignore")
        for raw in content.replace(",", "\n").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            w = normalize_wallet(line)
            if w:
                wallets.append(w)
    return sorted(set(wallets))


def audit_blacklist(output_dir: pathlib.Path, season: str, date_tag: str, removed_wallets: Iterable[str]) -> None:
    audit_path = output_dir / f"{season}-blacklist-audit-{date_tag}.csv"
    with audit_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wallet"])
        for w in sorted(set(removed_wallets)):
            writer.writerow([w])


def detect_total_runs(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    # returns (wallet_col, score_col)
    cols = {c.lower(): c for c in df.columns}
    wallet_candidates = ["walletaddress", "wallet", "address", "wallet_address", "player", "player_address"]
    score_candidates = ["total_runs", "score", "runs", "cumulative_run_score", "sum_score"]
    wallet_col = next((cols[c] for c in wallet_candidates if c in cols), None)
    score_col = next((cols[c] for c in score_candidates if c in cols), None)
    if wallet_col and score_col:
        return wallet_col, score_col
    return None


def detect_best_run(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    cols = {c.lower(): c for c in df.columns}
    wallet_candidates = ["walletaddress", "wallet", "address", "wallet_address", "player", "player_address"]
    score_candidates = ["best_run", "bestscore", "best_score", "high_score", "score"]
    wallet_col = next((cols[c] for c in wallet_candidates if c in cols), None)
    score_col = next((cols[c] for c in score_candidates if c in cols), None)
    if wallet_col and score_col:
        return wallet_col, score_col
    return None


def detect_credit_spend(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    cols = {c.lower(): c for c in df.columns}
    wallet_candidates = ["walletaddress", "wallet", "address", "wallet_address", "player", "player_address"]
    spend_candidates = ["creditspend", "credits_spent", "credits", "spend", "total_credits"]
    wallet_col = next((cols[c] for c in wallet_candidates if c in cols), None)
    spend_col = next((cols[c] for c in spend_candidates if c in cols), None)
    if wallet_col and spend_col:
        return wallet_col, spend_col
    return None


def detect_purchase_events(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    cols = {c.lower(): c for c in df.columns}
    wallet_candidates = ["address", "walletaddress", "wallet", "wallet_address", "player", "player_address"]
    eth_candidates = ["totalethether", "total_eth_ether", "eth_spent", "total_eth"]
    wallet_col = next((cols[c] for c in wallet_candidates if c in cols), None)
    eth_col = next((cols[c] for c in eth_candidates if c in cols), None)
    if wallet_col and eth_col:
        return wallet_col, eth_col
    return None


def load_weekly_inputs(input_dir: pathlib.Path, log_level: str) -> Dict[str, pd.DataFrame]:
    datasets: Dict[str, pd.DataFrame] = {}
    for csv_path in input_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        # Try detect each type
        if "total_runs" not in datasets:
            det = detect_total_runs(df)
            if det:
                datasets["total_runs"] = df
                log("debug", f"Detected total_runs: {csv_path.name}", log_level)
                continue
        if "best_run" not in datasets:
            det = detect_best_run(df)
            if det:
                datasets["best_run"] = df
                log("debug", f"Detected best_run: {csv_path.name}", log_level)
                continue
        if "credit_spend" not in datasets:
            det = detect_credit_spend(df)
            if det:
                datasets["credit_spend"] = df
                log("debug", f"Detected credit_spend: {csv_path.name}", log_level)
                continue
        if "purchase_events" not in datasets:
            det = detect_purchase_events(df)
            if det:
                datasets["purchase_events"] = df
                log("debug", f"Detected purchase_events: {csv_path.name}", log_level)
                continue
    return datasets


def df_to_leaderboard(df: pd.DataFrame, wallet_col: str, value_col: str, agg: str) -> LeaderboardData:
    wallet_lower_to_original: Dict[str, str] = {}
    value_by_wallet_lower: Dict[str, Decimal] = defaultdict(lambda: Decimal(0))
    for _, row in df.iterrows():
        wallet_orig = row.get(wallet_col)
        wallet_lower = normalize_wallet(wallet_orig)
        if not wallet_lower:
            continue
        if wallet_lower not in wallet_lower_to_original and isinstance(wallet_orig, str):
            wallet_lower_to_original[wallet_lower] = wallet_orig
        value = to_decimal(row.get(value_col))
        if value is None:
            continue
        if agg == "sum":
            value_by_wallet_lower[wallet_lower] += value
        elif agg == "max":
            if value > value_by_wallet_lower[wallet_lower]:
                value_by_wallet_lower[wallet_lower] = value
        else:
            raise ValueError("Unknown agg")
    return LeaderboardData(wallet_lower_to_original, value_by_wallet_lower)


def load_and_prepare(inputs: Dict[str, pd.DataFrame], min_credits: int) -> Tuple[LeaderboardData, LeaderboardData, Dict[str, Decimal], Dict[str, str]]:
    # Detect columns for each df
    total_wallet_col, total_score_col = detect_total_runs(inputs["total_runs"])  # type: ignore
    best_wallet_col, best_score_col = detect_best_run(inputs["best_run"])  # type: ignore
    spend_wallet_col, spend_col = detect_credit_spend(inputs["credit_spend"])  # type: ignore

    total_board = df_to_leaderboard(inputs["total_runs"], total_wallet_col, total_score_col, agg="sum")
    best_board = df_to_leaderboard(inputs["best_run"], best_wallet_col, best_score_col, agg="max")

    # Credits spent per wallet
    credits: Dict[str, Decimal] = {}
    for _, row in inputs["credit_spend"].iterrows():
        w = normalize_wallet(row.get(spend_wallet_col))
        v = to_decimal(row.get(spend_col))
        if w and v is not None:
            credits[w] = v

    # Unified map for preserving original case (prefer total, then best)
    wallet_case: Dict[str, str] = {}
    wallet_case.update(total_board.wallet_lower_to_original)
    for w, orig in best_board.wallet_lower_to_original.items():
        wallet_case.setdefault(w, orig)

    # Filter by min credits later within ranking/qualification step
    return total_board, best_board, credits, wallet_case


def load_purchase_events(inputs: Dict[str, pd.DataFrame]) -> Dict[str, Decimal]:
    if "purchase_events" not in inputs:
        return {}
    wallet_col, eth_col = detect_purchase_events(inputs["purchase_events"])  # type: ignore
    spend: Dict[str, Decimal] = defaultdict(lambda: Decimal(0))
    for _, row in inputs["purchase_events"].iterrows():
        w = normalize_wallet(row.get(wallet_col))
        v = to_decimal(row.get(eth_col))
        if w and v is not None:
            spend[w] += v
    return dict(spend)


def parse_eth_config(path: pathlib.Path) -> List[EthBand]:
    bands: List[EthBand] = []
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        leaderboard = str(row["leaderboard"]).strip().lower()
        rank_str = str(row["rank"]).strip().lower()
        winners = int(str(row["winners"]).strip())
        share_text = str(row["share"]).strip().replace("%", "")
        share_fraction = Decimal(share_text) / Decimal(100)
        if "-" in rank_str:
            a, b = rank_str.split("-", 1)
            # remove possible suffixes like 'th'
            a_num = int(''.join(ch for ch in a if ch.isdigit()))
            b_num = int(''.join(ch for ch in b if ch.isdigit()))
            start_rank, end_rank = a_num, b_num
        else:
            a_num = int(''.join(ch for ch in rank_str if ch.isdigit()))
            start_rank = a_num
            end_rank = a_num
        bands.append(EthBand(leaderboard=leaderboard, start_rank=start_rank, end_rank=end_rank, winners=winners, share_fraction=share_fraction))
    return bands


def parse_xp_config(path: pathlib.Path) -> Dict[str, Decimal]:
    weights: Dict[str, Decimal] = {}
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        leader = str(row["leaderboard"]).strip().lower()
        share_text = str(row["share"]).strip().replace("%", "")
        weights[leader] = Decimal(share_text) / Decimal(100)
    return weights


def split_eth_total(eth_total: Decimal, split_str: str) -> Tuple[Decimal, Decimal]:
    # format: "85:15" meaning total:best
    parts = split_str.split(":")
    if len(parts) != 2:
        raise ValueError("--eth-split must be like 85:15")
    total_pct = Decimal(parts[0])
    best_pct = Decimal(parts[1])
    if total_pct + best_pct != Decimal(100):
        # Normalize
        total_pct = (total_pct / (total_pct + best_pct)) * Decimal(100)
        best_pct = Decimal(100) - total_pct
    total_pool = (eth_total * total_pct / Decimal(100))
    best_pool = (eth_total * best_pct / Decimal(100))
    return total_pool, best_pool


def qualify_wallets(credits: Dict[str, Decimal], min_credits: int) -> set:
    return {w for w, c in credits.items() if c >= Decimal(min_credits)}


def rank_wallets(value_by_wallet: Dict[str, Decimal]) -> List[Tuple[str, Decimal]]:
    # Deterministic: by value desc, then wallet asc
    return sorted(value_by_wallet.items(), key=lambda kv: (-kv[1], kv[0]))


def distribute_eth_for_leaderboard(
    ranks: List[Tuple[str, Decimal]],
    pool: Decimal,
    bands: List[EthBand],
    target_leader: str,
    qualified: Optional[set] = None,
) -> Dict[str, Decimal]:
    payouts: Dict[str, Decimal] = defaultdict(lambda: Decimal(0))
    if pool <= 0:
        return payouts
    # Build map rank -> wallet
    rank_to_wallet: Dict[int, str] = {}
    for idx, (wallet, _) in enumerate(ranks, start=1):
        rank_to_wallet[idx] = wallet

    # First pass: collect wallets per band and active share sum
    active_bands: List[Tuple[EthBand, List[str]]] = []
    active_share = Decimal(0)
    for band in bands:
        if band.leaderboard != target_leader or band.share_fraction <= 0:
            continue
        recipients: List[str] = []
        for r in range(band.start_rank, band.end_rank + 1):
            w = rank_to_wallet.get(r)
            if w is not None:
                if qualified is None or w in qualified:
                    recipients.append(w)
        if recipients:
            active_bands.append((band, recipients))
            active_share += band.share_fraction

    if active_share <= 0:
        return payouts

    # Reallocate proportionally so that the entire pool is distributed among active bands
    for band, recipients in active_bands:
        normalized_share = band.share_fraction / active_share
        band_total = pool * normalized_share
        per_wallet = band_total / Decimal(len(recipients))
        for w in recipients:
            payouts[w] += per_wallet
    return payouts


def finalize_eth_payouts(payouts: Dict[str, Decimal], target_total: Decimal, eth_decimals: int, target_winners: Optional[int]) -> List[Tuple[str, Decimal, Decimal]]:
    # Returns list of (wallet_lower, unrounded_amount, rounded_amount)
    unit = Decimal(1) / (Decimal(10) ** eth_decimals)
    # Filter positive
    items = [(w, amt) for w, amt in payouts.items() if amt > 0]
    # Sort by amount desc, wallet asc for determinism
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    # Optional winner cap: keep top K and scale up to preserve total
    total_unrounded = sum(amt for _, amt in items)
    if target_winners is not None and len(items) > target_winners and total_unrounded > 0:
        kept = items[:target_winners]
        kept_sum = sum(amt for _, amt in kept)
        scale = (total_unrounded / kept_sum) if kept_sum > 0 else Decimal(1)
        items = [(w, amt * scale) for w, amt in kept]
        total_unrounded = sum(amt for _, amt in items)
    # Initial rounding
    rounded = [(w, amt, amt.quantize(unit, rounding=ROUND_HALF_UP)) for w, amt in items]
    sum_rounded = sum(r for _, _, r in rounded)
    # Reconcile rounding to match target total (quantized to unit)
    target_q = target_total.quantize(unit, rounding=ROUND_HALF_UP)
    diff = target_q - sum_rounded
    if diff != 0:
        steps = int((abs(diff) / unit).to_integral_value(rounding=ROUND_HALF_UP))
        # Order by fractional remainder (largest gets increment), or smallest for decrement
        def frac_part(x: Decimal) -> Decimal:
            # fractional relative to unit
            q = (x / unit)
            return q - q.to_integral_value(rounding=ROUND_HALF_UP) if diff > 0 else q - q.to_integral_value(rounding=ROUND_HALF_UP)
        # Precompute fractional distance to next unit
        ranked = list(rounded)
        if diff > 0:
            ranked.sort(key=lambda t: (-(t[1] / unit - (t[1] / unit).to_integral_value(rounding=ROUND_HALF_UP))), reverse=False)
            # Above sorting is tricky; simpler: sort by remainder descending
            ranked.sort(key=lambda t: (t[1] % unit) if unit != 0 else Decimal(0), reverse=True)
            for i in range(steps):
                idx = i % len(ranked)
                w, un, r = ranked[idx]
                ranked[idx] = (w, un, r + unit)
        else:
            # Reduce from the smallest fractional parts first; ensure not negative
            ranked.sort(key=lambda t: (t[1] % unit) if unit != 0 else Decimal(0))
            for i in range(steps):
                idx = i % len(ranked)
                w, un, r = ranked[idx]
                new_val = r - unit
                if new_val < 0:
                    # skip if would go negative; move to next
                    continue
                ranked[idx] = (w, un, new_val)
        rounded = ranked
    return rounded


def write_eth_output(output_dir: pathlib.Path, season: str, date_tag: str, payouts: Dict[str, Decimal], wallet_case: Dict[str, str], eth_decimals: int, target_total: Decimal, target_winners: Optional[int]) -> pathlib.Path:
    out_path = output_dir / f"{season}-eth-rewards-{date_tag}.csv"
    finalized = finalize_eth_payouts(payouts, target_total=target_total, eth_decimals=eth_decimals, target_winners=target_winners)
    rows: List[Tuple[str, str]] = []
    for w_lower, _un, rounded_amt in finalized:
        if rounded_amt <= 0:
            continue
        rows.append((wallet_case.get(w_lower, w_lower), str(rounded_amt)))
    # sort by payout desc for readability
    rows.sort(key=lambda x: Decimal(x[1]), reverse=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wallet", "payout"])
        writer.writerows(rows)
    return out_path


def write_xp_output(
    output_dir: pathlib.Path,
    season: str,
    date_tag: str,
    total_runs: LeaderboardData,
    best_run: LeaderboardData,
    spend_eth: Dict[str, Decimal],
    weights: Dict[str, Decimal],
    qualified_wallets: set,
    wallet_case: Dict[str, str],
) -> pathlib.Path:
    out_path = output_dir / f"{season}-xp-rewards-{date_tag}.csv"

    # Build union set of wallets that are qualified
    wallets = set(total_runs.value_by_wallet_lower.keys()) | set(best_run.value_by_wallet_lower.keys()) | set(spend_eth.keys())
    wallets &= qualified_wallets

    # Aggregates
    sum_total_runs = sum(total_runs.value_by_wallet_lower.get(w, Decimal(0)) for w in wallets)
    sum_best_run = sum(best_run.value_by_wallet_lower.get(w, Decimal(0)) for w in wallets)
    sum_spend = sum(spend_eth.get(w, Decimal(0)) for w in wallets)

    def safe_norm(val: Decimal, denom: Decimal) -> Decimal:
        return (val / denom) if denom > 0 else Decimal(0)

    rows: List[Tuple[int, str, str, str, str, str]] = []
    scores: List[Tuple[str, Decimal]] = []

    for w in wallets:
        v_total = total_runs.value_by_wallet_lower.get(w, Decimal(0))
        v_best = best_run.value_by_wallet_lower.get(w, Decimal(0))
        v_spend = spend_eth.get(w, Decimal(0))
        norm_total = safe_norm(v_total, sum_total_runs)
        norm_best = safe_norm(v_best, sum_best_run)
        norm_spend = safe_norm(v_spend, sum_spend)
        total_score = (
            weights.get("total", Decimal(0)) * norm_total
            + weights.get("best", Decimal(0)) * norm_best
            + weights.get("spend", Decimal(0)) * norm_spend
        )
        scores.append((w, total_score))

    # Rank by normalized total_score desc, tie-breaker by wallet asc
    scores.sort(key=lambda kv: (-kv[1], kv[0]))

    for rank_idx, (w, total_score) in enumerate(scores, start=1):
        v_total = total_runs.value_by_wallet_lower.get(w, Decimal(0))
        v_best = best_run.value_by_wallet_lower.get(w, Decimal(0))
        v_spend = spend_eth.get(w, Decimal(0))
        rows.append(
            (
                rank_idx,
                wallet_case.get(w, w),
                f"{v_total}",
                f"{v_best}",
                f"{v_spend}",
                # total_score should be normalized per user request
                f"{total_score}",
            )
        )

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "wallet_address", "cumulative_run_score", "best_run", "eth_spent", "total_score"])
        writer.writerows(rows)

    return out_path


def main() -> None:
    args = parse_args()
    log_level = args.log_level

    season = args.season
    input_dir = pathlib.Path(args.input_dir)
    config_dir = pathlib.Path(args.config_dir)
    output_dir = pathlib.Path(args.output_dir)
    date_tag = args.date or dt.datetime.now().strftime("%Y%m%d")

    # Load blacklist
    blacklist_wallets = set(read_blacklist(input_dir))

    # Load weekly datasets
    inputs = load_weekly_inputs(input_dir, log_level)
    missing = {k for k in ["total_runs", "best_run", "credit_spend"] if k not in inputs}
    if missing:
        raise SystemExit(f"Missing required inputs: {sorted(missing)}")

    # Convert to leaderboards and credits
    total_board, best_board, credits, wallet_case = load_and_prepare(inputs, args.min_credits)

    # Apply blacklist to leaderboards and credits
    def remove_blacklisted(board: LeaderboardData) -> LeaderboardData:
        filtered_values = {w: v for w, v in board.value_by_wallet_lower.items() if w not in blacklist_wallets}
        filtered_case = {w: o for w, o in board.wallet_lower_to_original.items() if w in filtered_values}
        return LeaderboardData(filtered_case, filtered_values)

    total_board = remove_blacklisted(total_board)
    best_board = remove_blacklisted(best_board)
    credits = {w: c for w, c in credits.items() if w not in blacklist_wallets}

    # Qualified wallets by credits (for payout eligibility only)
    qualified = qualify_wallets(credits, args.min_credits)

    # Rank leaderboards AFTER blacklist removal, but BEFORE credit qualification (no re-ranking by eligibility)
    total_ranks = rank_wallets(total_board.value_by_wallet_lower)
    best_ranks = rank_wallets(best_board.value_by_wallet_lower)

    # Parse ETH config
    eth_bands = parse_eth_config(config_dir / "arcade-eth-payout-structure.csv")

    # ETH distribution if eth-total provided
    eth_output_path = None
    if args.eth_total is not None:
        total_pool, best_pool = split_eth_total(args.eth_total, args.eth_split)
        total_payouts = distribute_eth_for_leaderboard(total_ranks, total_pool, eth_bands, target_leader="total", qualified=qualified)
        best_payouts = distribute_eth_for_leaderboard(best_ranks, best_pool, eth_bands, target_leader="best", qualified=qualified)
        # merge
        all_payouts: Dict[str, Decimal] = defaultdict(lambda: Decimal(0))
        for d in (total_payouts, best_payouts):
            for w, amt in d.items():
                all_payouts[w] += amt
        # Keep wallet_case updated for any wallet present only in payouts
        for w in list(all_payouts.keys()):
            wallet_case.setdefault(w, w)
        if not args.dry_run:
            eth_output_path = write_eth_output(output_dir, season, date_tag, all_payouts, wallet_case, args.eth_decimals, args.eth_total, args.eth_target_winners)

    # XP normalized output
    xp_weights = parse_xp_config(config_dir / "arcade-xp-payout-structure.csv")
    spend_eth = load_purchase_events(inputs)
    # Remove blacklisted from spend and re-qualify filtering
    spend_eth = {w: v for w, v in spend_eth.items() if w not in blacklist_wallets}
    # XP uses qualified wallets (credits + not blacklisted)

    xp_output_path = None
    if not args.dry_run:
        xp_output_path = write_xp_output(
            output_dir,
            season,
            date_tag,
            total_board,
            best_board,
            spend_eth,
            xp_weights,
            qualified,
            wallet_case,
        )

    # Blacklist audit
    if not args.dry_run:
        audit_blacklist(output_dir, season, date_tag, blacklist_wallets)

    # Reporting
    if eth_output_path:
        log("info", f"ETH rewards written: {eth_output_path}", log_level)
    log("info", f"XP normalized scores written: {xp_output_path}", log_level)


if __name__ == "__main__":
    main()
