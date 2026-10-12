# Copyright 2026 ETH Zurich and the relayout authors.
"""The examples the README shows run and check themselves."""

import pathlib
import runpy

EXAMPLES = pathlib.Path(__file__).parents[1] / "examples"


def test_the_omen_quickstart_runs() -> None:
    runpy.run_path(str(EXAMPLES / "omen_quickstart.py"), run_name="__main__")


if __name__ == "__main__":
    test_the_omen_quickstart_runs()
