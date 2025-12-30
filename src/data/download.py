"""Download ChoCo dataset."""

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

CHOCO_URL = "https://zenodo.org/api/records/7706751/files/smashub/choco-v1.0.0.zip?download=1"
CHOCO_FILENAME = "choco-v1.0.0.zip"


def download_choco(
    data_dir: Path | str = "data",
    force: bool = False,
) -> Path:
    """Download and extract ChoCo dataset.

    Args:
        data_dir: Directory to store the dataset.
        force: Whether to re-download if already exists.

    Returns:
        Path to the extracted dataset.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / CHOCO_FILENAME
    extract_dir = data_dir / "choco"

    # Check if already downloaded
    if extract_dir.exists() and not force:
        print(f"ChoCo already exists at {extract_dir}")
        return extract_dir

    # Download
    if not zip_path.exists() or force:
        print("Downloading ChoCo dataset (~467 MB)...")
        print(f"URL: {CHOCO_URL}")

        def progress_hook(block_num: int, block_size: int, total_size: int) -> None:
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 // total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\rProgress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")

        urlretrieve(CHOCO_URL, zip_path, reporthook=progress_hook)
        print("\nDownload complete!")

    # Extract
    print(f"Extracting to {extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    # The zip extracts to smashub/choco-v1.0.0, rename to choco
    extracted_path = data_dir / "smashub" / "choco-v1.0.0"
    if extracted_path.exists():
        if extract_dir.exists():
            import shutil

            shutil.rmtree(extract_dir)
        extracted_path.rename(extract_dir)
        # Clean up smashub dir if empty
        smashub_dir = data_dir / "smashub"
        if smashub_dir.exists() and not any(smashub_dir.iterdir()):
            smashub_dir.rmdir()

    print(f"Extraction complete! Dataset at {extract_dir}")
    return extract_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download ChoCo dataset")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    download_choco(args.data_dir, args.force)
