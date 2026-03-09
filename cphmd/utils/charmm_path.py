"""CHARMM-safe path quoting.

CHARMM's Fortran I/O (VOPEN) lowercases unquoted filenames.
On case-sensitive filesystems (Linux), this silently breaks any path
containing uppercase characters.  Wrapping the path in double quotes
tells the Fortran layer to preserve case.

Usage::

    from cphmd.utils.charmm_path import qpath

    read.psf_card(qpath(psf_file))
    lingo.charmm_script(f"stream {qpath(script)}")
    pycharmm.CharmmFile(file_name=qpath(log_path), ...)
"""

from pathlib import Path


def qpath(p: str | Path) -> str:
    """Return *p* as a double-quoted string for CHARMM Fortran I/O.

    Already-quoted strings are returned unchanged.
    """
    s = str(p)
    if s.startswith('"') and s.endswith('"'):
        return s
    return f'"{s}"'
