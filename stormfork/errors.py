from typing import Literal


class UserError(Exception):
    """A CLI failure with one actionable next step and a documented exit code."""

    def __init__(self, what: str, next_step: str, exit_code: Literal[1, 2, 3]) -> None:
        super().__init__(what)
        self.what = what
        self.next_step = next_step
        self.exit_code = exit_code
