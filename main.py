from __future__ import annotations

import time
from pathlib import Path


class AgentCLI:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parent
        self.steps_dir = self.root / "steps"

    def run(self) -> int:
        self._print_menu()
        choice = input("\nEnter a, b, c, or d: ").strip().lower()
        return self._dispatch(choice)

    def _print_menu(self) -> None:
        print("Choose an option:")
        print("(a) Find the (x, y) coordinates/pixels of a described UI element")
        print("(b) Extract text from the screen")
        print("(c) Determine the next action (click/keystroke/scroll + location)")
        print("(d) Scroll and find the (x, y) coordinates of row(s) in a table")

    def _open_screen(self) -> None:
        print("\nPlease open/focus the screen you want to automate.")
        delay_s = 5
        print(f"Starting in {delay_s}s...")
        time.sleep(delay_s)
        print("Starting now.\n")

    def _dispatch(self, choice: str) -> int:

        if choice == "a":
            return self._option_find_coordinates()
        if choice == "b":
            return self._option_extract_text()
        if choice == "c":
            return self._option_next_action()
        if choice == "d":
            return self._option_scroll_find_rows()

        print("Invalid choice. Please enter a, b, c, or d.")
        return 1

    def _option_find_coordinates(self) -> int:
        description = input("Describe the UI element (e.g. 'Chrome icon'): ").strip()
        print(f"Description: {description}")
        self._open_screen()
        from steps.find_cordinates import FindCoordinatesStep

        FindCoordinatesStep(description=description, out_dir=Path("out")).run()
        return 0

    def _option_extract_text(self) -> int:
        description = input("Describe the text you want to extract: ").strip()
        self._open_screen()
        from steps.extract_text import ExtractTextStep

        step = ExtractTextStep(user_task=description, target_count=None, out_dir=Path("out"))
        result = step.run()
        print(result)
        return 0

    def _option_next_action(self) -> int:
        goal = input("Describe what you want to do (e.g. 'open Chrome and search cats'): ").strip()
        self._open_screen()
        from steps.next_action import NextActionStep

        NextActionStep(goal=goal, out_dir=Path("out")).run()
        return 0

    def _option_scroll_find_rows(self) -> int:
        row_desc = input("Which row(s) to find?: ").strip()
        from steps.scroll_find_rows import ScrollFindRowsStep

        ScrollFindRowsStep(value_query=row_desc, out_dir=Path("out")).run()
        return 0


def main() -> None:
    # Plain call: run the menu and return normally.
    # If you want to propagate the numeric exit code, you can print it or handle it here.
    AgentCLI().run()


if __name__ == "__main__":
    main()

