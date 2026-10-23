#!/usr/bin/env python3
"""Compatibility entry point; implementation is shared with test_ai.py."""
import sys

from test_ai import main


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if not any(arg in arguments for arg in ("--image", "--camera", "--url")):
        arguments.extend(("--camera", "0"))
    sys.argv = [sys.argv[0]] + arguments + ["--mode", "landmark"]
    raise SystemExit(main())
