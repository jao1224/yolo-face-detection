"""
Validação Formal do Modelo YOLO treinado no conjunto de teste/validação
"""

import os
from ultralytics import YOLO

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Prioridade para o modelo em models/best.pt ou runs/...
    model_path = os.path.join(base_dir, 'models', 'best.pt')
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, 'runs', 'detect', 'train-14', 'weights', 'best.pt')
        
    data_yaml = os.path.join(base_dir, 'data.yaml')
    
    print(f"--> Carregando modelo: {model_path}")
    model = YOLO(model_path)
    
    print(f"--> Executando validação oficial (model.val) no dataset ({data_yaml})...")
    output_dir = os.path.join(base_dir, 'runs', 'detect', 'val_results')
    
    metrics = model.val(
        data=data_yaml,
        split='val',
        imgsz=512,
        batch=16,
        project=output_dir,
        name='val_train14',
        exist_ok=True
    )
    
    print("\n================ RESULTADOS DA VALIDAÇÃO ================")
    print(f"mAP@0.5:        {metrics.box.map50:.4f} ({metrics.box.map50*100:.2f}%)")
    print(f"mAP@0.5:0.95:   {metrics.box.map:.4f} ({metrics.box.map*100:.2f}%)")
    print(f"Precisão (P):   {metrics.box.mp:.4f} ({metrics.box.mp*100:.2f}%)")
    print(f"Recall (R):     {metrics.box.mr:.4f} ({metrics.box.mr*100:.2f}%)")
    f1 = 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-16)
    print(f"F1-Score:       {f1:.4f} ({f1*100:.2f}%)")
    print("=========================================================")

if __name__ == "__main__":
    main()
