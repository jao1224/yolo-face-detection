"""
Treinamento / Fine-Tuning do Modelo YOLO para Detecção Facial
Dataset: WiderFace | Resolução Otimizada: 512px
"""

import os
from ultralytics import YOLO

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Pesos de inicialização (pode usar yolo26n.pt, yolov8n.pt ou checkpoint anterior)
    initial_weights = os.path.join(base_dir, 'models', 'best.pt')
    if not os.path.exists(initial_weights):
        initial_weights = 'yolov8n.pt'  # fallback automático para modelo base
    
    data_yaml = os.path.join(base_dir, 'data.yaml')
    
    print(f"--> Inicializando modelo a partir de: {initial_weights}")
    print(f"--> Arquivo de dados: {data_yaml}")
    
    model = YOLO(initial_weights)

    # Hiperparâmetros otimizados para detecção facial de alta precisão e sensibilidade
    model.train(
        data=data_yaml,
        epochs=1000,
        imgsz=512,          # 512px para preservar resolução de rostos pequenos e distantes
        batch=8,            # Tamanho de lote estável
        resume=False,       # Iniciar novo ciclo refinando os pesos
        patience=300,       # Early stopping com margem ampla
        cos_lr=True,        # Decaimento cosseno da taxa de aprendizado
        cls=0.7,            # Peso aumentado na perda de classe para maximizar Recall
        box=10.0,           # Peso aumentado na perda de caixa para contorno facial preciso
        dfl=2.0,            # Foco em bordas e detalhes finos
        workers=0,          # Compatibilidade total com Windows
        project=os.path.join(base_dir, 'runs', 'detect'),
        name='train_custom',
        exist_ok=True
    )

if __name__ == "__main__":
    main()
