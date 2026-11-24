#!/usr/bin/env python3
"""Validate the manually selected Homebrew release tag."""
import argparse
from pathlib import Path
import re
import sys


def validate_tag(tag):
    number = r'(?:0|[1-9][0-9]*)'
    if not isinstance(tag, str) or not re.fullmatch(rf'v{number}\.{number}\.{number}(?:-[0-9A-Za-z.-]+)?', tag):
        raise ValueError('expected a release version tag')
    if '-' in tag:
        for part in tag.split('-', 1)[1].split('.'):
            if not part or (part.isdigit() and len(part) > 1 and part[0] == '0'):
                raise ValueError('invalid prerelease identifier')
    return tag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True, help='explicit manual Homebrew selection')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tag = validate_tag(args.tag)
    with args.output.open('a') as output:
        output.write(f'tag={tag}\n')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as error:
        sys.exit(f'Homebrew release selection failed: {error}')
