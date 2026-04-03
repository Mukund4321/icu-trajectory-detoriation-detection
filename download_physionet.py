"""
PhysioNet Challenge 2019 Downloader
=====================================
Downloads the Sepsis Early Prediction dataset (training_setA) from PhysioNet.
Files land in:  data/raw/training_setA/*.psv  (one .psv file per patient)

Usage:
    python download_physionet.py              # downloads first 2000 patients (quick test)
    python download_physionet.py --all        # downloads all ~20,336 patients (full dataset)
    python download_physionet.py --n 5000     # downloads first N patients

Dataset info:
    Each .psv file = one ICU patient, one row per hour, ~40 features + SepsisLabel
    training_setA: 20,336 patients
    Reference: https://physionet.org/content/challenge-2019/1.0.0/
"""

import argparse
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL  = "https://physionet.org/files/challenge-2019/1.0.0/training/training_setA"
TOTAL_SET_A = 20336
DEFAULT_N   = 2000   # Enough for a solid IEEE result; downloads in ~5 minutes

def download_one(patient_num: int, out_dir: Path) -> tuple:
    """Download a single patient file. Returns (patient_num, success, error_msg)."""
    filename = f"p{patient_num:06d}.psv"
    url      = f"{BASE_URL}/{filename}"
    dest     = out_dir / filename

    if dest.exists():
        return patient_num, True, "cached"

    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, dest)
            return patient_num, True, None
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return patient_num, False, "404"
            time.sleep(0.5 * (attempt + 1))
        except Exception as e:
            time.sleep(0.5 * (attempt + 1))

    return patient_num, False, "failed after 3 retries"


def main():
    parser = argparse.ArgumentParser(description="Download PhysioNet Challenge 2019 data")
    parser.add_argument("--all", action="store_true", help="Download all patients in set A")
    parser.add_argument("--n",   type=int, default=DEFAULT_N, help="Number of patients to download")
    args = parser.parse_args()

    n_patients = TOTAL_SET_A if args.all else args.n

    out_dir = Path("data/raw/training_setA")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check how many are already downloaded
    existing = len(list(out_dir.glob("*.psv")))
    if existing >= n_patients:
        print(f"OK {existing} patient files already in {out_dir} — nothing to download.")
        print("  Run  python main.py  to start the pipeline.")
        return

    print(f"Downloading {n_patients} patients from PhysioNet Challenge 2019...")
    print(f"  Output dir : {out_dir.resolve()}")
    print(f"  Already had: {existing} files")
    print(f"  To download: {n_patients - existing} files")
    print(f"  Workers    : 20 parallel threads\n")

    patient_nums = list(range(0, n_patients))
    success, skipped, failed = 0, 0, 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(download_one, num, out_dir): num for num in patient_nums}
        for i, future in enumerate(as_completed(futures), 1):
            num, ok, msg = future.result()
            if ok:
                if msg == "cached":
                    skipped += 1
                else:
                    success += 1
            else:
                if msg != "404":
                    failed += 1

            if i % 100 == 0 or i == len(patient_nums):
                elapsed = time.time() - start
                rate = i / elapsed
                remaining = (len(patient_nums) - i) / rate if rate > 0 else 0
                total_ok = success + skipped
                print(f"  [{i:>6}/{len(patient_nums)}]  "
                      f"downloaded={success}  cached={skipped}  "
                      f"failed={failed}  "
                      f"speed={rate:.0f}/s  "
                      f"ETA={remaining:.0f}s")

    elapsed = time.time() - start
    total_files = len(list(out_dir.glob("*.psv")))

    print(f"\n{'='*60}")
    print(f"Download complete in {elapsed:.0f}s")
    print(f"  Total .psv files in {out_dir}: {total_files}")
    print(f"{'='*60}")

    if total_files > 0:
        print(f"\nOK Ready. Run the pipeline with:")
        print(f"    python main.py")
    else:
        print("\nFAIL No files downloaded. Check your internet connection.")
        sys.exit(1)


if __name__ == "__main__":
    main()
