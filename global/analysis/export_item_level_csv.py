#!/usr/bin/env python3
"""Export the item-level CSV (2296 rows) cited by the paper's RQ1/RQ2.

Writes analysis/output/item_level_all_methods.csv with one row per
(item, method): item_id, system, method, CtxTok, Ctx.R, Faith, Ans.R,
Ans.C, RA-FC.
"""
from common import OUTPUT, load_results


def main() -> None:
    df = load_results()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT / "item_level_all_methods.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
