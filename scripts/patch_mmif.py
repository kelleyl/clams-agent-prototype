#!/usr/bin/env python3
"""Patch mmif-python to handle missing cross-view references gracefully.

The mmif SDK eagerly resolves annotation cross-references during deserialization,
which crashes with KeyError when views reference annotations from other views
that haven't been fully indexed. This patch:

1. Makes __getitem__ return None instead of raising KeyError for missing IDs
2. Wraps _get_linear_anchor_point calls in try/except to catch cascading errors
"""
import sys

path = sys.argv[1]
with open(path) as f:
    lines = f.readlines()

patched = []
i = 0
while i < len(lines):
    line = lines[i]

    # Patch 1: __getitem__ KeyError -> return None
    if "raise KeyError" in line and "not found in the MMIF object" in line:
        indent = len(line) - len(line.lstrip())
        patched.append(" " * indent + "return None  # patched: was KeyError\n")
        i += 1
        continue

    # Patch 2: Wrap _get_linear_anchor_point calls in try/except
    if "_props_ephemeral" in line and "_get_linear_anchor_point" in line:
        indent = len(line) - len(line.lstrip())
        spaces = " " * indent
        patched.append(spaces + "try:\n")
        patched.append(spaces + "    " + line.lstrip())
        if "start=True" in line:
            fallback = "    ann._props_ephemeral['start'] = None\n"
        else:
            fallback = "    ann._props_ephemeral['end'] = None\n"
        patched.append(spaces + "except (KeyError, AttributeError, TypeError, ValueError):\n")
        patched.append(spaces + fallback)
        i += 1
        continue

    patched.append(line)
    i += 1

with open(path, "w") as f:
    f.writelines(patched)

n_none = sum(1 for l in patched if "was KeyError" in l)
n_try = sum(1 for l in patched if "except (KeyError, AttributeError" in l)
print(f"Patched {path}: {n_none} KeyError->None, {n_try} try/except blocks")
