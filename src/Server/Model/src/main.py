import os
import sys
import argparse
from mega_detector_system import MegaDeepFake, Trainer


def main():
    parser = argparse.ArgumentParser(description="MegaDeepFake Detector CLI")
    parser.add_argument(
        "video", type=str,
        help="Path to the video file for DeepFake scoring"
    )
    parser.add_argument(
        "--fast-validate", action="store_true",
        help="Run quick validation on the training/validation subsets"
    )
    args = parser.parse_args()

    if args.video:
        if not os.path.isfile(args.video):
            print(f"Error: File '{args.video}' not found.", file=sys.stderr)
            sys.exit(1)
        score = MegaDeepFake().predict(args.video)
        is_fake = score > 0.5
        label = "FAKE" if is_fake else "REAL"
        print(f"DeepFake Score: {score:.4f} → {label}")
        sys.exit(1 if is_fake else 0)

    if args.fast_validate:
        Trainer().run()
        sys.exit(0)

    parser.print_help()
    sys.exit(0)


if __name__ == "__main__":
    main()