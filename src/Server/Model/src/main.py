import sys
import os
import torch
from detector import TwoStreamDeepFakeDetector, preprocess_two_stream

def get_executable_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def main():
    if len(sys.argv) != 2:
        print("❌ Ошибка: укажите путь к видео.")
        sys.exit(2)

    video_path = sys.argv[1]
    if not os.path.isfile(video_path):
        print(f"❌ Видео не найдено: {video_path}")
        sys.exit(3)

    exe_dir    = get_executable_dir()
    model_path = os.path.abspath(os.path.join(exe_dir, '..', 'models', 'deepfake_detector.pth'))
    if not os.path.isfile(model_path):
        print(f"❌ Вес модели не найдены по пути: {model_path}")
        sys.exit(4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Устройство: {device}")

    print("🔄 Загружаем модель...")
    model = TwoStreamDeepFakeDetector().to(device)
    ckpt  = torch.load(model_path, map_location=device)
    state_dict = ckpt.get('model_state', ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    print("🎞️ Обрабатываем видео (RGB + Flow)...")
    rgb_t, flow_t = preprocess_two_stream(video_path)
    rgb_t, flow_t = rgb_t.to(device), flow_t.to(device)

    print("🤖 Выполняем инференс...")
    with torch.no_grad():
        logit = model(rgb_t, flow_t)
        prob  = torch.sigmoid(logit).item()

    print(f"📊 Вероятность дипфейка: {prob:.4f}")
    label = 1 if prob > 0.5 else 0
    print(f"🏷️ Результат: {'DeepFake' if label else 'Real'}")
    sys.exit(label)

if __name__ == "__main__":
    main()
