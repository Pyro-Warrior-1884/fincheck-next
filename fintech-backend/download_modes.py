import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

RELEASE_TAG = "v1-models"
BASE_URL = f"https://github.com/mukesh1352/fincheck-next/releases/download/{RELEASE_TAG}"

MODELS = [
    "baseline_mnist.pth",
    "kd_mnist.pth",
    "lrf_mnist.pth",
    "pruned_mnist.pth",
    "quantized_mnist.pth",
    "ws_mnist.pth",
]

def ensure_models():
    missing = []

    for name in MODELS:
        if not (MODEL_DIR / name).exists():
            missing.append(name)

    if not missing:
        print("✅ All model files already present")
        return

    print(f"⬇️ Downloading {len(missing)} missing model(s)")

    for name in missing:
        url = f"{BASE_URL}/{name}"
        dest = MODEL_DIR / name

        print(f"⬇️ {name}")
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()

        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

    print("🎉 Model download complete")
