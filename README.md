## Arcade Rewards CSV Processing

Python project to calculate and distribute ETH rewards and XP scores for Arcade game seasons.

### Features
- **ETH Rewards**: Distributes ETH to top performers on Total Runs and Best Run leaderboards
- **XP Scoring**: Calculates normalized XP scores based on Total Runs, Best Run, and ETH Spent
- **Flexible Eligibility**: Multiple eligibility modes for XP rewards
- **Auto-scaling**: Automatic score scaling for readable XP values
- **Blacklist Support**: Removes blacklisted wallets from all rewards

### Structure
- `config/`: CSV configuration files for payout structures
- `weekly-input/`: Weekly raw input CSVs (total_runs, best_run, credit_spend, purchase_events)
- `calculate_rewards.py`: Main rewards calculation script

### Setup
1. Create and activate a virtual environment
```bash
python3 -m venv .venv && source .venv/bin/activate
```
2. Install dependencies
```bash
pip install -r requirements.txt
```

### Usage

#### Basic Example
```bash
python calculate_rewards.py --season season10 --eth-total 3.47
```

#### Full Example with Options
```bash
python calculate_rewards.py \
  --season season10 \
  --eth-total 3.47 \
  --eth-split 85:15 \
  --min-credits 20 \
  --xp-eligibility hybrid \
  --xp-scale auto \
  --input-dir weekly-input \
  --config-dir config \
  --output-dir .
```

### Parameters

#### Required
- `--season`: Season identifier (e.g., `season10`, `s10`)
- `--eth-total`: Total ETH to distribute across leaderboards

#### Optional
- `--eth-split`: Split between total:best leaderboards (default: `85:15`)
- `--min-credits`: Minimum credits spent to qualify for ETH rewards (default: `20`)
- `--xp-eligibility`: XP eligibility mode (default: `min-credits`)
  - `min-credits`: Require min-credits threshold
  - `hybrid`: min-credits OR (0.001+ ETH spent with 1+ credit)
  - `all`: Include all wallets
- `--xp-scale`: XP score multiplier (default: `auto`)
  - `auto`: Auto-scale so minimum score = 100
  - `<number>`: Fixed multiplier (e.g., `10000`, `100000`)
- `--eth-decimals`: Decimal places for ETH output (default: `18`)
- `--eth-target-winners`: Optional target number of ETH winners (post-merge)
- `--input-dir`: Weekly input directory (default: `weekly-input`)
- `--config-dir`: Config directory (default: `config`)
- `--output-dir`: Output directory (default: `.`)
- `--date`: YYYYMMDD override for output filenames (default: today)
- `--dry-run`: Compute but do not write files
- `--log-level`: Log verbosity: `debug`, `info`, `warn` (default: `info`)

### Input Files

The script expects the following CSVs in `weekly-input/`:
- **Total Runs**: Must have wallet and total_runs/score columns
- **Best Run**: Must have wallet and best_run/best_score columns
- **Credit Spend**: Must have wallet and credits_spent columns
- **Purchase Events** (optional): Must have wallet and total_eth_ether columns
- **Blacklist** (optional): `.txt` file with one wallet address per line

### Output Files

The script generates three output files:
1. `{season}-eth-rewards-{date}.csv`: ETH payouts per wallet
2. `{season}-xp-rewards-{date}.csv`: XP scores with rankings
3. `{season}-blacklist-audit-{date}.csv`: List of removed wallets

#### XP Rewards Columns
- `rank`: Final ranking by total_score
- `wallet_address`: Player wallet address
- `cumulative_run_score`: Total runs score
- `best_run`: Best single run score
- `eth_spent`: Total ETH spent on purchases
- `credits_used`: Total credits used
- `total_score`: Normalized and scaled XP score

### XP Calculation

XP scores are calculated using weighted normalization:
1. Each metric (total_runs, best_run, eth_spent) is normalized proportionally across eligible wallets
2. Normalized values are weighted according to `config/arcade-xp-payout-structure.csv`
3. Final scores are scaled by `--xp-scale` multiplier (default: auto-scales to min=100)

### Notes
- `weekly-input/` is ignored by Git to keep raw data out of the repo
- ETH rewards use band-based distribution with automatic reallocation
- Wallets are ranked deterministically (by score desc, then wallet asc)
