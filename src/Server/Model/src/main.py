import os
import sys
import argparse
from mega_detector_system import MegaDeepFake, Trainer


def main():
    parser = argparse.ArgumentParser(description="MegaDeepFake Detector CLI")
    parser.add_argument(
        "--video", type=str,
        help="Path to the video file for DeepFake scoring"
    )
    parser.add_argument(
        "--fast-validate", action="store_true",
        help="Run quick validation on the training/validation subsets"
    )
    parser.add_argument(
        "--download-models-only", action="store_true",
        help="Download all required models without running detection"
    )
    args = parser.parse_args()

    # Просто скачать модели и выйти
    if args.download_models_only:
        print("⬇️ Downloading models for MegaDeepFake...")
        MegaDeepFake().warmup()  # метод warmup должен внутри вызвать загрузку всех моделей
        print("✅ All models downloaded.")
        sys.exit(0)

    # Быстрая проверка (валидация)
    if args.fast_validate:
        print("🧪 Running fast validation...")
        Trainer().run()
        sys.exit(0)

    # Предсказание по видео
    if args.video:
        if not os.path.isfile(args.video):
            print(f"❌ Error: File '{args.video}' not found.", file=sys.stderr)
            sys.exit(1)
        print(f"🎬 Analyzing video: {args.video}")
        score = MegaDeepFake().predict(args.video)
        is_fake = score > 0.5
        label = "FAKE" if is_fake else "REAL"
        print(f"✅ DeepFake Score: {score:.4f} → {label}")
        sys.exit(1 if is_fake else 0)

    # Если ничего не передано
    parser.print_help()
    sys.exit(0)


if __name__ == "__main__":
    main()