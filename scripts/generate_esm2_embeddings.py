#!/usr/bin/env python3
"""Generate ESM2-650M mean-pooled embeddings for the three ProteinGym assays.

Reproduces the logic from MLS-Bench vendor/data_scripts/ProteinGym/prepare_data.py
but as a standalone script usable with a plain Python venv (fair-esm + torch).

Outputs (one .pt per assay, matching validate_official_data expectations):
    <data_root>/proteingym_embeddings/{BLAT_ECOLX_Firnberg_2014,ESTA_BACSU_Nutschel_2020,RASH_HUMAN_Bandaru_2017}.pt
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch


ASSAYS = [
    "BLAT_ECOLX_Firnberg_2014",
    "ESTA_BACSU_Nutschel_2020",
    "RASH_HUMAN_Bandaru_2017",
]
EMBED_DIM = 1280
BATCH_SIZE = 16


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Directory containing proteingym/DMS_substitutions.csv and proteingym/DMS_assays/")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    import esm

    pg = args.data_root / "proteingym"
    ref_path = pg / "DMS_substitutions.csv"
    assay_root = pg / "DMS_assays"
    embed_root = args.data_root / "proteingym_embeddings"
    embed_root.mkdir(parents=True, exist_ok=True)

    if not ref_path.exists():
        raise SystemExit(f"reference CSV missing: {ref_path}")
    if not assay_root.exists():
        raise SystemExit(f"DMS assay dir missing: {assay_root}")

    print(f"loading ESM2-650M on {args.device}", flush=True)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    device = torch.device(args.device)
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    with ref_path.open() as fh:
        ref = {row["DMS_id"]: row for row in csv.DictReader(fh)}

    for assay in ASSAYS:
        out = embed_root / f"{assay}.pt"
        if out.exists():
            print(f"SKIP {assay} (exists)", flush=True)
            continue
        row = ref[assay]
        target_seq = row["target_seq"]
        csv_files = sorted(assay_root.rglob(f"{assay}*.csv"))
        if not csv_files:
            raise RuntimeError(f"No DMS CSV for {assay}")
        import pandas as pd

        df = pd.read_csv(csv_files[0])
        # Accept both the official schema (mutant/mutated_sequence/DMS_score)
        # and the converted schema from convert_proteingym_dms.py.
        if "mutated_sequence" in df.columns and "DMS_score" in df.columns:
            df = df[~df["mutant"].astype(str).str.contains(":", regex=False)].reset_index(drop=True)
            seqs = [str(target_seq)]
            seq_to_idx = {str(target_seq): 0}
            for _, r in df.iterrows():
                ms = str(r["mutated_sequence"])
                if ms not in seq_to_idx:
                    seq_to_idx[ms] = len(seqs)
                    seqs.append(ms)
        else:
            # Official ProteinGym DMS csv: mutant/mutated_sequence/score columns
            df = df[~df["mutant"].astype(str).str.contains(":", regex=False)].reset_index(drop=True)
            seqs = [str(target_seq)]
            seq_to_idx = {str(target_seq): 0}
            for _, r in df.iterrows():
                ms = str(r["mutated_sequence"])
                if ms not in seq_to_idx:
                    seq_to_idx[ms] = len(seqs)
                    seqs.append(ms)

        all_emb = torch.zeros(len(seqs), EMBED_DIM)
        t0 = __import__("time").time()
        with torch.no_grad():
            for i in range(0, len(seqs), BATCH_SIZE):
                batch = seqs[i : i + BATCH_SIZE]
                data = [(str(j), seq) for j, seq in enumerate(batch)]
                _, _, batch_tokens = batch_converter(data)
                batch_tokens = batch_tokens.to(device)
                out_tok = model(batch_tokens, repr_layers=[33])["representations"][33]
                # mean-pool over non-pad tokens
                mask = batch_tokens != alphabet.padding_idx
                emb = (out_tok * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
                all_emb[i : i + BATCH_SIZE] = emb.cpu()
                print(f"  {assay} {min(i + BATCH_SIZE, len(seqs))}/{len(seqs)} "
                      f"({__import__('time').time() - t0:.0f}s)", flush=True)

        scores = df["DMS_score"].astype(float).to_numpy()
        # Embeddings are indexed by unique sequence (seqs[1:] in df order);
        # keep one score per unique mutated sequence in the same order.
        unique_scores = []
        unique_mutant_ids = []
        for _, r in df.iterrows():
            ms = str(r["mutated_sequence"])
            idx = seq_to_idx[ms]
            if idx == len(unique_scores) + 1:  # first time this seq is seen (wt=0)
                unique_scores.append(float(r["DMS_score"]))
                unique_mutant_ids.append(str(r["mutant"]))
        scores_t = torch.tensor(unique_scores, dtype=torch.float32)
        wt_embedding = all_emb[0]
        mutant_ids = unique_mutant_ids

        payload = {
            "embeddings": all_emb[1:].to(torch.float32),
            "scores": scores_t,
            "wt_embedding": wt_embedding.to(torch.float32),
            "mutant_ids": mutant_ids,
        }
        assert payload["embeddings"].shape[0] == payload["scores"].shape[0] == len(mutant_ids), (
            f"{assay} mismatch: emb={payload['embeddings'].shape[0]} scores={payload['scores'].shape[0]} mutants={len(mutant_ids)}"
        )
        torch.save(payload, out)
        print(f"wrote {out} shape={payload['embeddings'].shape} ({__import__('time').time() - t0:.0f}s)", flush=True)

    print("ALL_EMBEDDINGS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
