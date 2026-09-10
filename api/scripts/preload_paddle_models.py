"""Download the exact Paddle models used by dense CAD OCR during image build."""

import os
from pathlib import Path


MODEL_NAMES = (
    "PP-OCRv5_mobile_det",
    "th_PP-OCRv5_mobile_rec",
)


def main() -> int:
    # The production host is in mainland China. ModelScope carries both exact
    # OCR packages and avoids waiting for unavailable Hugging Face/AiStudio
    # fallbacks during every clean image build.
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "modelscope")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddlex.inference.utils.official_models import official_models

    for model_name in MODEL_NAMES:
        model_directory = Path(official_models[model_name])
        if not model_directory.is_dir():
            raise RuntimeError(f"Paddle model was not downloaded: {model_name}")
        print(f"preloaded {model_name}: {model_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
