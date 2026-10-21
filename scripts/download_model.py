"""Download and verify the documented SpaceNet 2 DeepLab checkpoint."""
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/"models"/"best_deeplab.pth"
TEMP=DEST.with_suffix(".pth.part")
URL="https://huggingface.co/Abhatta7/building-footprint-segmentation/resolve/e34a637/best_deeplab.pth"
EXPECTED_SIZE=168_357_171
EXPECTED_SHA256="170c49e1f42737f5a4b3922b9fc58b25b6ba0a7dd7bf00b9186a92ccb84dc555"

def main():
    DEST.parent.mkdir(parents=True,exist_ok=True)
    digest=sha256(); size=0
    try:
        with urlopen(URL,timeout=60) as response, TEMP.open("wb") as out:
            while chunk:=response.read(1024*1024):
                out.write(chunk); digest.update(chunk); size+=len(chunk)
    except Exception:
        TEMP.unlink(missing_ok=True); raise
    if size!=EXPECTED_SIZE or digest.hexdigest()!=EXPECTED_SHA256:
        TEMP.unlink(missing_ok=True)
        raise RuntimeError(f"Checkpoint verification failed (bytes={size}, sha256={digest.hexdigest()})")
    TEMP.replace(DEST)
    print(f"Verified model checkpoint: {DEST} ({size:,} bytes)")

if __name__=="__main__": main()
