#!/usr/bin/env python3
"""Replace local absolute paths in delimited files with their basenames."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path, PureWindowsPath


WINDOWS_PATH = re.compile(r"^[A-Za-z]:\\")


def sanitize(value: str) -> str:
    if WINDOWS_PATH.match(value):
        return PureWindowsPath(value).name
    if value.startswith("/Users/") or value.startswith("/home/"):
        return Path(value).name
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument(
        "--delimiter",
        choices=("auto", "comma", "tab"),
        default="auto",
        help="Delimiter selection; auto uses csv.Sniffer.",
    )
    args = parser.parse_args()

    for path in args.files:
        text = path.read_text(encoding="utf-8-sig")
        if args.delimiter == "comma":
            delimiter = ","
        elif args.delimiter == "tab":
            delimiter = "\t"
        else:
            delimiter = csv.Sniffer().sniff(text[:65536], delimiters=",\t").delimiter

        rows = list(csv.reader(text.splitlines(), delimiter=delimiter))
        changed = 0
        for row in rows:
            for index, value in enumerate(row):
                clean_value = sanitize(value)
                if clean_value != value:
                    row[index] = clean_value
                    changed += 1

        with path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle, delimiter=delimiter, lineterminator="\n").writerows(rows)
        print(f"Sanitized {changed} path values in {path}")


if __name__ == "__main__":
    main()
