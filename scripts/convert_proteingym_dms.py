#!/usr/bin/env python3
"""Convert ProteinGYM-DMS HF tsv files into ai4bio mutation-task CSV assets.

The ai4bio task needs, for each of its three assays:
  * an assay DMS csv with columns `mutant`, `mutated_sequence`, `DMS_score`
    (used by the ESM2 embedding generator), and
  * a cv_folds csv named `{assay_id}*.csv` with columns `mutant`,
    `fold_random_5` (validated by validate_official_data).

The genbio-ai/ProteinGYM-DMS HF dataset stores the same data as tsv with
columns `sequences`, `labels`, `fold_id`; this script converts them.

Usage:
  python scripts/convert_proteingym_dms.py <src_tsv_dir> <out_root> <wt_fasta_dir>
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd


ASSAY_IDS = {
    "BLAT_ECOLX": ("BLAT_ECOLX_Firnberg_2014", "BLAT_ECOLX_Firnberg_2014.tsv"),
    "ESTA_BACSU": ("ESTA_BACSU_Nutschel_2020", "ESTA_BACSU_Nutschel_2020.tsv"),
    "RASH_HUMAN": ("RASH_HUMAN_Bandaru_2017", "RASH_HUMAN_Bandaru_2017.tsv"),
}

# Wild-type sequences for the three assays (target_seq from ProteinGym).
WT_SEQUENCES = {
    "BLAT_ECOLX": (
        "MSIQHFRVALIPFFAAFCLPVFASPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"
    ),
    "ESTA_BACSU": (
        "MPMVAGTCLLLALLAALPAGSVHALTLSSESNAGFGSPEVTQVDASSPVTELVDRWGGVVVKSGAVRQMLENYAEKAGLPFVAVHCDVHEVLTWFTAGLDKGHFVNMVTDLAQAGFDARVIAIGHSTGGGVVQAAANETRIPRVVVMAGGVDEWGPLGVPGAIVQGIPVGMALANQAYEPNAGWPELWGSFPEVVKELLSDGSVDYIVNVPHNPPMVAGGTKVAALGSSPTTYPTGYSVALGGYGNKDAPQLLKASGQALVRYVNLGRGAGVAVQVGGGHSTTALQEWLQGEGLEKLGMRQFDWTAWSTTVSLGKTTGFVLFDPAAAGSIANAAQALQAYEKVAGATVSIGSVVYGNADYGANSAVDGQLTRWVSEEYPANAPAVSEAGAVVGSLLVSGGGGGTVASFGKVSAYSDTTAGDQVTAVWVGVLGK"
    ),
    "RASH_HUMAN": (
        "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMSCKCVLS"
    ),
}


def derive_mutant(wt: str, seq: str) -> str:
    """Derive the ProteinGym single-substitution mutant label from wt vs seq."""
    diffs = [(i, wt[i], seq[i]) for i in range(min(len(wt), len(seq))) if wt[i] != seq[i]]
    if len(diffs) == 0:
        return wt[0] + str(1) + wt[0]  # placeholder for wild-type row
    if len(diffs) == 1:
        i, a, b = diffs[0]
        return f"{a}{i + 1}{b}"
    # multi-site or indel; join with colon like ProteinGym
    return ":".join(f"{a}{i + 1}{b}" for i, a, b in diffs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src_dir", type=Path, help="Directory with the downloaded *tsv files.")
    parser.add_argument("out_root", type=Path, help="Output root (proteingym/ layout will be created).")
    args = parser.parse_args()

    assay_dir = args.out_root / "proteingym" / "DMS_assays" / "DMS_ProteinGym_substitutions"
    cv_dir = args.out_root / "proteingym" / "cv_folds" / "cv_folds_singles_substitutions"
    assay_dir.mkdir(parents=True, exist_ok=True)
    cv_dir.mkdir(parents=True, exist_ok=True)

    for short, (assay_id, fname) in ASSAY_IDS.items():
        src = args.src_dir / fname
        if not src.exists():
            print(f"MISSING {src}")
            continue
        df = pd.read_csv(src, sep="\t")
        # Drop rows containing ':' (multi-site variants) to match ProteinGym singles.
        df = df[~df["sequences"].astype(str).str.contains(":", regex=False)].reset_index(drop=True)
        wt = WT_SEQUENCES[short]
        df["mutant"] = [derive_mutant(wt, str(s)) for s in df["sequences"]]
        df["mutated_sequence"] = df["sequences"].astype(str)
        df["DMS_score"] = df["labels"].astype(float)

        # Assay DMS csv (embedding generator input).
        assay_csv = assay_dir / f"{assay_id}.csv"
        df[["mutant", "mutated_sequence", "DMS_score"]].to_csv(assay_csv, index=False)
        print(f"wrote {assay_csv} ({len(df)} rows)")

        # cv_folds csv: mutant + fold_random_5 (fold_id is 0-4 already).
        cv_csv = cv_dir / f"{assay_id}.csv"
        folds = pd.DataFrame({
            "mutant": df["mutant"].astype(str),
            "fold_random_5": df["fold_id"].astype(int),
        })
        folds.to_csv(cv_csv, index=False)
        print(f"wrote {cv_csv} ({len(folds)} rows, folds={sorted(folds['fold_random_5'].unique())})")

    # DMS_substitutions.csv reference (needed by embedding script ref lookup).
    print("\nNOTE: also write DMS_substitutions.csv reference rows if the embedding script uses DMS_id lookup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
