#!/usr/bin/env python3
"""
DEPRECATED: This file is kept for backward compatibility.

Please use train.py instead:
    python APG/train.py

Or as a module:
    python -m APG.train
"""

import warnings

warnings.warn(
    "apg_run.py is deprecated. Please use train.py instead: python APG/train.py",
    DeprecationWarning,
    stacklevel=2,
)

from APG.train import main

if __name__ == "__main__":
    main()
