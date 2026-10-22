"""Download and verify the published BIT change detector trained on LEVIR-CD."""
from hashlib import sha256
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models" / "BIT_LEVIR.onnx"
URL = "https://huggingface.co/LEEhyeongsuk/BIT-LEVIR-CD/resolve/main/BIT_LEVIR.onnx"
EXPECTED_SIZE = 12_638_542
EXPECTED_SHA256 = "edf413687700d765eef6e6af6b7d7dc101738707255ea6e112898a1f4e52af9b"


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = DEST.with_suffix(".onnx.download")
    if not DEST.exists():
        print(f"Downloading LEVIR-CD BIT weights from {URL}")
        urlretrieve(URL, temporary)
        temporary.replace(DEST)
    size = DEST.stat().st_size
    digest = sha256(DEST.read_bytes()).hexdigest()
    if size != EXPECTED_SIZE or digest != EXPECTED_SHA256:
        DEST.unlink(missing_ok=True)
        raise SystemExit(f"Checkpoint verification failed: size={size}, SHA-256={digest}")
    print(f"Verified {DEST} ({size:,} bytes; SHA-256 {digest})")


if __name__ == "__main__":
    main()
