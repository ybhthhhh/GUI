"""Exercise a real browser GUI path suitable for the Mem-W web stage.

This checks browser launch, page rendering, text entry, navigation, and an
optional screenshot.  It is an infrastructure check, not a benchmark score:
the public website may change and no task-success metric is inferred here.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://www.wikipedia.org/")
    parser.add_argument("--query", default="Web automation")
    parser.add_argument("--proxy", help="Playwright proxy URL, e.g. http://127.0.0.1:7897")
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, proxy={"server": args.proxy} if args.proxy else None)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            page.goto(args.url, wait_until="domcontentloaded", timeout=60_000)
            search = page.locator("input[name=search]")
            search.fill(args.query)
            search.press("Enter")
            page.wait_for_load_state("domcontentloaded", timeout=60_000)
            if args.screenshot:
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(args.screenshot), full_page=False)
            print("MEMW_WEB_GUI_INTERACTION_OK", page.url, page.title(), "search_input_used=1")
        finally:
            browser.close()


if __name__ == "__main__":
    main()
