"""Entry point for python -m silent_bug_agent."""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
