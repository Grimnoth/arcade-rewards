## Arcade Rewards CSV Processing

Python project to process CSV payout structures for Arcade Rewards.

### Structure
- `config/`: CSV configuration files
- `weekly-input/`: Weekly raw inputs (ignored by Git)

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
Add your processing script(s) and run them, for example:
```bash
python process_csvs.py
```

### Notes
- `weekly-input/` is ignored by Git to keep raw data out of the repo.
