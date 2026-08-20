"""Create roll-number-only PDF, MP4, and ZIP submission artifacts."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROLL_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roll-number", required=True)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submission" / "final")
    args = parser.parse_args()

    roll = args.roll_number.strip()
    if not ROLL_PATTERN.fullmatch(roll):
        raise SystemExit("Roll number may contain only letters, numbers, hyphens, and underscores")
    video = args.video.resolve()
    if not video.is_file() or video.suffix.lower() != ".mp4":
        raise SystemExit("--video must point to an existing MP4 recording")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{roll}.pdf"
    video_path = output_dir / f"{roll}.mp4"
    zip_path = output_dir / f"{roll}.zip"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_submission_deck.py"),
            "--roll-number",
            roll,
            "--output",
            str(pdf_path),
        ],
        cwd=ROOT,
        check=True,
    )
    if video != video_path:
        shutil.copy2(video, video_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(pdf_path, pdf_path.name)
        archive.write(video_path, video_path.name)

    with zipfile.ZipFile(zip_path) as archive:
        expected = {pdf_path.name, video_path.name}
        if set(archive.namelist()) != expected or archive.testzip() is not None:
            raise SystemExit("Submission ZIP validation failed")
    print(f"Created {pdf_path}")
    print(f"Created {video_path}")
    print(f"Created {zip_path}")


if __name__ == "__main__":
    main()
