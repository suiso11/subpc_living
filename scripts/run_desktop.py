import sys
from pathlib import Path


def _main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    from src.desktop.app import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_main())
