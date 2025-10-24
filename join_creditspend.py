import argparse
import csv
import pathlib
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "weekly-input"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join credit spend data to leaderboard CSVs")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Input directory containing CSV files")
    return parser.parse_args()


def normalize_wallet(addr: object) -> Optional[str]:
    if addr is None:
        return None
    s = str(addr).strip()
    if s == "":
        return None
    return s.lower()


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


def find_file_by_patterns(input_dir: pathlib.Path, patterns: List[str]) -> Optional[pathlib.Path]:
    """Find first file matching any of the given patterns in filename."""
    for csv_path in sorted(input_dir.glob("*.csv")):
        filename_lower = csv_path.name.lower()
        # Skip generated/joined files
        if "join" in filename_lower or "audit" in filename_lower or "rewards" in filename_lower:
            continue
        for pattern in patterns:
            if pattern in filename_lower:
                return csv_path
    return None


def load_weekly_inputs(input_dir: pathlib.Path) -> Dict[str, pd.DataFrame]:
    """Load weekly input files with explicit file type handling."""
    datasets: Dict[str, pd.DataFrame] = {}
    
    # Define explicit file patterns for each dataset type
    file_specs = [
        ("total_runs", ["total_run", "total-run", "cumulative"], detect_total_runs),
        ("best_run", ["best_run", "best-run", "bestrun", "best_score"], detect_best_run),
        ("credit_spend", ["credit", "spend"], detect_credit_spend),
    ]
    
    # First pass: explicit filename-based detection
    for dataset_name, patterns, validator_func in file_specs:
        if dataset_name in datasets:
            continue
            
        csv_path = find_file_by_patterns(input_dir, patterns)
        if csv_path:
            try:
                df = pd.read_csv(csv_path)
                # Validate columns match expected structure
                if validator_func(df):
                    datasets[dataset_name] = df
                    print(f"✓ Loaded {dataset_name}: {csv_path.name}")
                else:
                    print(f"⚠ File {csv_path.name} matched pattern but has unexpected columns for {dataset_name}")
            except Exception as e:
                print(f"✗ Failed to load {csv_path.name}: {e}")
    
    return datasets


def join_credit_spend_to_leaderboard(
    leaderboard_df: pd.DataFrame,
    credit_df: pd.DataFrame,
    lb_wallet_col: str,
    credit_wallet_col: str,
    credit_col: str,
    output_path: pathlib.Path
) -> None:
    """Join credit spend data to leaderboard and write output."""
    # Create credit spend lookup (lowercase wallet -> credits)
    credit_lookup: Dict[str, int] = {}
    for _, row in credit_df.iterrows():
        wallet = normalize_wallet(row.get(credit_wallet_col))
        credits = row.get(credit_col)
        if wallet and credits is not None:
            try:
                credit_lookup[wallet] = int(credits)
            except (ValueError, TypeError):
                credit_lookup[wallet] = 0
    
    # Add credits_spent column to leaderboard
    credits_spent_list = []
    for _, row in leaderboard_df.iterrows():
        wallet = normalize_wallet(row.get(lb_wallet_col))
        credits = credit_lookup.get(wallet, 0)
        credits_spent_list.append(credits)
    
    # Create output dataframe with credits_spent column
    output_df = leaderboard_df.copy()
    output_df['credits_spent'] = credits_spent_list
    
    # Write to CSV
    output_df.to_csv(output_path, index=False)
    print(f"✓ Created: {output_path.name}")


def main() -> None:
    args = parse_args()
    input_dir = pathlib.Path(args.input_dir)
    
    if not input_dir.exists():
        print(f"✗ Error: Input directory does not exist: {input_dir}")
        return
    
    print(f"\nLoading files from: {input_dir}\n")
    
    # Load weekly datasets
    inputs = load_weekly_inputs(input_dir)
    missing = {k for k in ["total_runs", "best_run", "credit_spend"] if k not in inputs}
    if missing:
        print(f"\n✗ Error: Missing required inputs: {sorted(missing)}")
        return
    
    # Get column names
    total_cols = detect_total_runs(inputs["total_runs"])
    best_cols = detect_best_run(inputs["best_run"])
    credit_cols = detect_credit_spend(inputs["credit_spend"])
    
    if not total_cols or not best_cols or not credit_cols:
        print("\n✗ Error: Could not detect required columns")
        return
    
    total_wallet_col, _ = total_cols
    best_wallet_col, _ = best_cols
    credit_wallet_col, credit_col = credit_cols
    
    print("\nJoining credit spend data...\n")
    
    # Find original filenames to create output names
    total_runs_file = find_file_by_patterns(input_dir, ["total_run", "total-run", "cumulative"])
    best_run_file = find_file_by_patterns(input_dir, ["best_run", "best-run", "bestrun"])
    
    if total_runs_file:
        output_name = total_runs_file.stem + "_join_creditspend.csv"
        output_path = input_dir / output_name
        join_credit_spend_to_leaderboard(
            inputs["total_runs"],
            inputs["credit_spend"],
            total_wallet_col,
            credit_wallet_col,
            credit_col,
            output_path
        )
    
    if best_run_file:
        output_name = best_run_file.stem + "_join_creditspend.csv"
        output_path = input_dir / output_name
        join_credit_spend_to_leaderboard(
            inputs["best_run"],
            inputs["credit_spend"],
            best_wallet_col,
            credit_wallet_col,
            credit_col,
            output_path
        )
    
    print("\n✓ Done!\n")


if __name__ == "__main__":
    main()

