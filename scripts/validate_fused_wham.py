#!/usr/bin/env python3
"""Validate fused WHAM produces identical C/V output to serial WHAM.

Usage:
    python scripts/validate_fused_wham.py <analysis_dir> [--nsubs 3] [--temp 298.15]

Runs WHAM twice on the same input data:
  1. CPHMD_WHAM_SERIAL=1 (legacy serial dlnZ + custom get_CC)
  2. CPHMD_WHAM_SERIAL=0 (fused dlnZ + cuBLAS DGEMM)
Then compares C.dat and V.dat element-wise.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


def run_wham_in_dir(
    analysis_dir: Path,
    output_dir: Path,
    nsubs: list[int],
    temp: float,
    serial: bool,
    g_imp_path: str | None,
    gpu_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run WHAM from packed data and return C.dat, V.dat arrays."""
    # Set serial toggle
    os.environ["CPHMD_WHAM_SERIAL"] = "1" if serial else "0"

    from cphmd.core.alf_utils import compute_wham_inputs
    from cphmd.wham import run_wham_from_memory

    lambda_arrays, energy_matrix, gshift_data, nf = compute_wham_inputs(
        analysis_dir,
        nsubs=nsubs,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    run_wham_from_memory(
        lambda_arrays=lambda_arrays,
        energy_matrix=energy_matrix,
        nf=nf,
        temp=temp,
        nsubs=nsubs,
        use_gshift=g_imp_path is not None,
        g_imp_path=g_imp_path or "",
        gshift_data=gshift_data,
        output_dir=str(output_dir),
        gpu_id=gpu_id,
    )

    C = np.loadtxt(output_dir / "C.dat")
    V = np.loadtxt(output_dir / "V.dat")
    return C, V


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_dir", type=Path, help="Analysis directory with lambda/energy data")
    parser.add_argument("--nsubs", type=int, nargs="+", default=[3], help="Subsites per site")
    parser.add_argument("--temp", type=float, default=298.15)
    parser.add_argument("--g-imp-path", type=str, default=None)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=1e-8, help="Relative tolerance for comparison")
    parser.add_argument("--atol", type=float, default=1e-12, help="Absolute tolerance for comparison")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="wham_validate_") as tmpdir:
        serial_dir = Path(tmpdir) / "serial"
        fused_dir = Path(tmpdir) / "fused"

        print("Running WHAM (serial)...")
        C_serial, V_serial = run_wham_in_dir(
            args.analysis_dir, serial_dir, args.nsubs, args.temp,
            serial=True, g_imp_path=args.g_imp_path, gpu_id=args.gpu_id,
        )

        print("Running WHAM (fused + DGEMM)...")
        C_fused, V_fused = run_wham_in_dir(
            args.analysis_dir, fused_dir, args.nsubs, args.temp,
            serial=False, g_imp_path=args.g_imp_path, gpu_id=args.gpu_id,
        )

    # Compare
    print(f"\nC.dat shape: serial={C_serial.shape}, fused={C_fused.shape}")
    print(f"V.dat shape: serial={V_serial.shape}, fused={V_fused.shape}")

    c_maxdiff = np.max(np.abs(C_serial - C_fused))
    v_maxdiff = np.max(np.abs(V_serial - V_fused))
    print(f"C.dat max abs diff: {c_maxdiff:.2e}")
    print(f"V.dat max abs diff: {v_maxdiff:.2e}")

    try:
        np.testing.assert_allclose(C_serial, C_fused, rtol=args.rtol, atol=args.atol)
        print("C.dat: PASS")
    except AssertionError as e:
        print(f"C.dat: FAIL\n{e}")

    try:
        np.testing.assert_allclose(V_serial, V_fused, rtol=args.rtol, atol=args.atol)
        print("V.dat: PASS")
    except AssertionError as e:
        print(f"V.dat: FAIL\n{e}")

    if c_maxdiff < args.atol and v_maxdiff < args.atol:
        print("\nVALIDATION PASSED: fused path matches serial path.")
        sys.exit(0)
    else:
        print("\nVALIDATION FAILED: differences exceed tolerance.")
        sys.exit(1)


if __name__ == "__main__":
    main()
