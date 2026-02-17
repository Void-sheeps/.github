import asyncio
from playwright.async_api import async_playwright
import os

async def verify_terminal():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        # Load the local index.html
        path = os.path.abspath("index.html")
        await page.goto(f"file://{path}")

        print("Waiting for terminal to be visible...")
        # Terminal appears after a delay (simulateDownloadProcess starts after 1s, then 300ms)
        await page.wait_for_selector("#terminal", state="visible", timeout=10000)

        print("Waiting for specific logs...")
        # We'll wait for one of our new logs to appear
        logs_to_check = [
            "narrative_immersion.py --simulate",
            "narrative_analysis.py --analyze",
            "narrative_dynamics.py --simulate-dynamics",
            "hex_engine.py --snapshot",
            "hex_analysis.py --visualize",
            "Initializing CloverPit Hex Grid Engine"
        ]

        found_logs = []

        # Check periodically for logs
        for _ in range(30): # Wait up to ~15 seconds
            content = await page.inner_text("#terminal-content")
            status_text = await page.inner_text("#status-text")

            for log in logs_to_check:
                if (log in content or log in status_text) and log not in found_logs:
                    print(f"✓ Found: {log}")
                    found_logs.append(log)

            if len(found_logs) >= 4: # If we find most of them, we're good
                break

            await asyncio.sleep(0.5)

        await page.screenshot(path="terminal_verification.png")
        print("Screenshot saved to terminal_verification.png")

        if len(found_logs) < 2:
            print("Failed to find enough logs!")
            exit(1)

        await browser.close()

if __name__ == "__main__":
    asyncio.run(verify_terminal())
