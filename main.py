"""
Real-time Face Detection using Fine-Tuned YOLOv26 / YOLOv8 Nano
Dataset: WiderFace | Optimized Model: train-14 (512px)
"""

import os
import time
import cv2
from ultralytics import YOLO

def find_model_path():
    env_model = os.getenv('MODEL_PATH')
    if env_model and os.path.exists(env_model):
        return env_model

    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, 'models', 'best.pt'),
        os.path.join(base_dir, 'runs', 'detect', 'train-14', 'weights', 'best.pt'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Não foi possível encontrar os pesos 'best.pt' em 'models/' ou 'runs/'.")

def main():
    modelo_path = find_model_path()
    print(f"--> Carregando modelo YOLO de: {modelo_path}")
    model = YOLO(modelo_path)

    # Parâmetros configuráveis via variáveis de ambiente (.env) ou padrão
    webcam_idx = int(os.getenv('WEBCAM_INDEX', '0'))
    conf_thresh = float(os.getenv('CONFIDENCE_THRESHOLD', '0.25'))

    # Inicializar webcam
    print(f"--> Abrindo webcam (índice {webcam_idx})... (Pressione 'q' para sair)")
    webcam = cv2.VideoCapture(webcam_idx)
    if not webcam.isOpened():
        print(f"Erro: Não foi possível acessar a webcam no índice {webcam_idx}.")
        return

    prev_time = time.time()

    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Erro: Falha ao capturar quadro da webcam.")
            break

        # Cálculo de FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Executar inferência
        results = model(frame, conf=conf_thresh, verbose=False)

        num_faces = 0
        for result in results:
            for box in result.boxes:
                num_faces += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                # Desenhar caixa delimitadora e rótulo
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Face {conf:.2f}"
                cv2.putText(
                    frame, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
                )

        # Informações de status na tela
        cv2.putText(
            frame, f"FPS: {fps:.1f} | Rostos: {num_faces}", (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2
        )

        # Exibir resultado
        cv2.imshow("YOLO Real-Time Face Detection (Q to exit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()
    print("Execução finalizada com sucesso.")

if __name__ == "__main__":
    main()
