"""
download_sample.py
==================
Downloads the default sperm whale audio sample from Wikimedia Commons.

Run this ONCE before your first analysis:
    python download_sample.py

What it downloads:
    File    : Sperm_Whale_Normal_Clicks.ogg
    Species : Physeter macrocephalus (Sperm Whale)
    Source  : Wikimedia Commons
    URL     : https://upload.wikimedia.org/wikipedia/commons/3/36/Sperm_Whale_Normal_Clicks.ogg
    License : CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0/)

    The recording contains normal echolocation clicks of a sperm whale.
    It is publicly available for educational and research use under the
    Creative Commons Attribution-ShareAlike 3.0 licence.

    You must credit: Wikimedia Commons contributors / original recordist.

If the automatic download fails (e.g. network issues), this script will
print the manual download URL so you can save the file yourself.
"""

import os
import sys
import urllib.request
import urllib.error

# Primary and fallback sources
SOURCES = [
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/2/29/Sperm_Whale_Normal_Clicks.ogg",
        "filename": "sperm_whale_clicks.ogg",
        "description": "Sperm Whale Normal Clicks (Wikimedia Commons, Public Domain / NOAA)",
    },
]

OUTPUT_DIR = "data"


def download(url: str, dest_path: str) -> bool:
    """
    Download a file from url to dest_path.
    Returns True on success, False on failure.
    Prints progress every 256 KB.
    """
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Whale-Vocalization-Analysis-Project/1.0; "
                    "educational use)"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            total = int(response.headers.get("Content-Length", 0))
            chunk_size = 16 * 1024  # 16 KB chunks
            downloaded = 0
            with open(dest_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        bar = int(pct // 5) * "#"
                        print(
                            f"\r  [{bar:<20}] {pct:5.1f}%  ({downloaded:,} / {total:,} bytes)",
                            end="",
                            flush=True,
                        )
            print()  # newline after progress bar
        return True

    except urllib.error.URLError as exc:
        print(f"\n  [NETWORK ERROR] {exc.reason}")
        return False
    except OSError as exc:
        print(f"\n  [FILE ERROR] {exc}")
        return False


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for source in SOURCES:
        dest = os.path.join(OUTPUT_DIR, source["filename"])

        if os.path.isfile(dest):
            size = os.path.getsize(dest)
            print(f"[OK] File already exists: {dest}  ({size:,} bytes)")
            print("     Delete it and re-run this script if you want a fresh copy.")
            return True

        print(f"\nDownloading: {source['description']}")
        print(f"  From : {source['url']}")
        print(f"  To   : {dest}")

        success = download(source["url"], dest)

        if success:
            size = os.path.getsize(dest)
            print(f"\n[SUCCESS] Saved to {dest}  ({size:,} bytes)")
            print("\nYou can now run the analysis:")
            print("    python whale_analysis.py")
            return True
        else:
            # Clean up incomplete file
            if os.path.exists(dest):
                os.remove(dest)

    # All sources failed
    print(
        "\n[MANUAL DOWNLOAD REQUIRED]\n"
        "Automatic download failed. Please download the file manually:\n"
        "\n"
        "  1. Open this URL in your browser:\n"
        "     https://upload.wikimedia.org/wikipedia/commons/3/36/Sperm_Whale_Normal_Clicks.ogg\n"
        "\n"
        "  2. Save the file as:\n"
        f"     {os.path.abspath(os.path.join(OUTPUT_DIR, 'sperm_whale_clicks.ogg'))}\n"
        "\n"
        "  Then run:\n"
        "     python whale_analysis.py\n"
        "\n"
        "  Alternative: use your own WAV recording:\n"
        "     python whale_analysis.py path/to/your_recording.wav\n"
    )
    return False


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
