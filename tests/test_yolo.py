"""
Script de Teste de Inferência em Lote para Imagens Estáticas
"""

import os
from ultralytics import YOLO

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Caminho do modelo
    model_path = os.path.join(base_dir, 'models', 'best.pt')
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, 'runs', 'detect', 'train-14', 'weights', 'best.pt')
    
    print(f"--> Carregando modelo: {model_path}")
    model = YOLO(model_path)

    # Imagens de teste na pasta tests/
    test_dir = os.path.join(base_dir, 'tests')
    test_images = []
    for img_name in ['hotel1.jpg', 'hotel2.jpg', 'test_security_cam_frame.jpg', 'test_hotel_frame.jpg']:
        full_path = os.path.join(test_dir, img_name)
        if os.path.exists(full_path):
            test_images.append(full_path)

    if not test_images:
        print("Nenhuma imagem de teste encontrada em tests/.")
        return

    print(f"--> Executando inferência em {len(test_images)} imagem(ns)...")
    
    # Criar pasta de saída
    output_dir = os.path.join(base_dir, 'tests', 'test_results')
    os.makedirs(output_dir, exist_ok=True)

    results = model.predict(
        source=test_images,
        conf=0.25,
        save=True,
        project=output_dir,
        name='predict_test',
        exist_ok=True
    )

    for res in results:
        img_path = res.path
        num_faces = len(res.boxes)
        print(f"\n[Resultado] {os.path.basename(img_path)}:")
        print(f"  - Rostos detectados: {num_faces}")
        for j, box in enumerate(res.boxes):
            conf = float(box.conf[0])
            xyxy = [round(x, 1) for x in box.xyxy[0].tolist()]
            print(f"    * Face #{j+1}: Confiança = {conf:.2%}, Coordenadas [x1, y1, x2, y2] = {xyxy}")

    print(f"\n--> Imagens com detecções salvas com sucesso em: {os.path.join(output_dir, 'predict_test')}")

if __name__ == "__main__":
    main()
