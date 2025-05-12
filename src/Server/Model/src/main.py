import sys
from mega_detector_system import MegaDeepFake

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_video>")
        sys.exit(2)  # 2 = неправильное использование

    video_path = sys.argv[1]
    detector = MegaDeepFake()
    score = detector.predict(video_path)

    is_fake = int(score > 0.5)
    print(f"Score: {score:.3f} → {'FAKE' if is_fake else 'REAL'}")
    sys.exit(is_fake)

if __name__ == '__main__':
    main()
# main.py

import sys
from mega_detector_system import MegaDeepFake

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_video>")
        sys.exit(2)  # 2 = неправильное использование

    video_path = sys.argv[1]
    detector = MegaDeepFake()
    score = detector.predict(video_path)

    is_fake = int(score > 0.5)
    print(f"Score: {score:.3f} → {'FAKE' if is_fake else 'REAL'}")
    sys.exit(is_fake)

if __name__ == '__main__':
    main()
