"""Download PPGPT_500k_iters.pth from upstream HeartGPT (harryjdavies/HeartGPT)."""
import sys
import urllib.request
import zipfile
from pathlib import Path

UPSTREAM_ZIP = (
    "https://raw.githubusercontent.com/harryjdavies/HeartGPT/main/"
    "Model_files/PPGPT_500k_iters.zip"
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "HeartGPT" / "Model_files"
    out_dir.mkdir(parents=True, exist_ok=True)
    pth = out_dir / "PPGPT_500k_iters.pth"
    if pth.exists():
        print(f"Already present: {pth}")
        return

    zip_path = out_dir / "PPGPT_500k_iters.zip"
    print(f"Downloading {UPSTREAM_ZIP} …")
    urllib.request.urlretrieve(UPSTREAM_ZIP, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    zip_path.unlink(missing_ok=True)
    if not pth.exists():
        print(f"[ERROR] Expected {pth} after extract", file=sys.stderr)
        sys.exit(1)
    print(f"OK: {pth}")


if __name__ == "__main__":
    main()
