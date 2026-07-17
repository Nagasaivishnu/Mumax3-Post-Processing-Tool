"""
OVF2 Reader
============
Minimal reader for MuMax3 OVF2 binary files (float32 and float64).

MuMax3 always writes OVF2 Binary-4 (float32).  The format is:

  # OOMMF OVF 2.0
  ...header lines...
  # Begin: Data Binary 4
  [4-byte IEEE float = 1234567.0   ← endianness check]
  [nx * ny * nz * valuedim floats, x varies fastest, z slowest]
  # End: Data Binary 4

Returned array shape: (nz, ny, nx, valuedim)
where axis-0 is z (slowest in file), axis-2 is x (fastest in file).
"""

from __future__ import annotations
import numpy as np


def read_ovf(path: str) -> np.ndarray:
    """
    Read one MuMax3 OVF2 file.

    Parameters
    ----------
    path : path to the .ovf file

    Returns
    -------
    np.ndarray, shape (nz, ny, nx, valuedim), dtype float32

    Raises
    ------
    ValueError  if the file format is unexpected
    IOError     if the file cannot be read
    """
    with open(path, "rb") as fh:
        header: dict[str, str] = {}
        dtype  = None
        word   = 0

        # ── parse text header ─────────────────────────────────────────
        while True:
            raw  = fh.readline()
            line = raw.decode("latin-1", errors="replace").strip()

            lower = line.lower()
            if lower.startswith("# begin: data binary 4"):
                dtype, word = np.float32, 4
                break
            elif lower.startswith("# begin: data binary 8"):
                dtype, word = np.float64, 8
                break
            elif lower.startswith("# begin: data text"):
                raise ValueError(
                    "OVF text format is not supported; "
                    "set MuMax3 to output binary: OutputFormat = OVF2_TEXT is slow, "
                    "use the default binary output."
                )

            # collect key: value pairs
            if line.startswith("#") and ":" in line:
                key, _, val = line[1:].partition(":")
                header[key.strip().lower()] = val.strip()

        # ── validate required header fields ───────────────────────────
        for required in ("xnodes", "ynodes", "znodes"):
            if required not in header:
                raise ValueError(f"OVF header missing field: '{required}'")

        nx   = int(header["xnodes"])
        ny   = int(header["ynodes"])
        nz   = int(header["znodes"])
        vdim = int(header.get("valuedim", 3))

        # ── verify endianness / check value ───────────────────────────
        check_bytes = fh.read(word)
        check_val   = np.frombuffer(check_bytes, dtype=dtype)[0]
        if abs(float(check_val) - 1_234_567.0) > 0.5:
            raise ValueError(
                f"OVF check value mismatch: expected 1234567.0, got {check_val:.1f}. "
                "File may be corrupted or use unexpected byte order."
            )

        # ── read binary data ──────────────────────────────────────────
        n_values = nx * ny * nz * vdim
        n_bytes  = n_values * word
        raw_data = fh.read(n_bytes)
        if len(raw_data) != n_bytes:
            raise IOError(
                f"Unexpected end of file in '{path}': "
                f"expected {n_bytes} bytes, got {len(raw_data)}."
            )

        data = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)

    # Reshape: OVF stores z slowest, x fastest → (nz, ny, nx, vdim)
    return data.reshape(nz, ny, nx, vdim)
