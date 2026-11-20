#!/usr/bin/env python3
"""
Converte um CSV numérico (ex.: Covertype da UCI) para o formato binário
lido pelas versões em C/CUDA do K-means:

    int32 N | int32 D | N*D float32 (row-major)

Exemplos:
  # Covertype (54 atributos + 1 coluna de classe no fim, que é descartada)
  python3 scripts/prepare_dataset.py data/covtype.data.gz data/covtype.bin --drop-last

  # CSV com cabeçalho
  python3 scripts/prepare_dataset.py dados.csv data/dados.bin --skip-header

  # Dados sintéticos só para testar o pipeline (NÃO vale como base real)
  python3 scripts/prepare_dataset.py --synthetic 200000 16 data/teste.bin
"""
import argparse
import gzip
import sys

import numpy as np


def open_text(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", help="CSV de entrada (.csv ou .gz)")
    ap.add_argument("output", help="arquivo .bin de saída")
    ap.add_argument("--drop-last", action="store_true",
                    help="descarta a última coluna (rótulo/classe)")
    ap.add_argument("--skip-header", action="store_true",
                    help="ignora a primeira linha do CSV")
    ap.add_argument("--no-normalize", action="store_true",
                    help="não aplica z-score (por padrão aplica)")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="usa só as primeiras N linhas")
    ap.add_argument("--synthetic", nargs=2, type=int, metavar=("N", "D"),
                    help="gera dados sintéticos em vez de ler um CSV")
    args = ap.parse_args()

    if args.synthetic:
        n, d = args.synthetic
        rng = np.random.default_rng(42)
        centers = rng.normal(0, 5, size=(8, d))
        X = centers[rng.integers(0, 8, size=n)] + rng.normal(0, 1, size=(n, d))
    else:
        if not args.input:
            sys.exit("Informe o CSV de entrada ou use --synthetic N D")
        with open_text(args.input) as f:
            X = np.loadtxt(f, delimiter=",", dtype=np.float64,
                           skiprows=1 if args.skip_header else 0,
                           max_rows=args.max_rows)
        if args.drop_last:
            X = X[:, :-1]

    # Normalização z-score: sem ela, atributos com escala grande
    # (ex.: elevação em metros) dominariam a distância euclidiana.
    if not args.no_normalize:
        std = X.std(axis=0)
        std[std == 0] = 1.0
        X = (X - X.mean(axis=0)) / std

    X = np.ascontiguousarray(X, dtype=np.float32)
    n, d = X.shape
    with open(args.output, "wb") as f:
        np.array([n, d], dtype=np.int32).tofile(f)
        X.tofile(f)
    print(f"OK: {args.output}  N={n}  D={d}  ({X.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
