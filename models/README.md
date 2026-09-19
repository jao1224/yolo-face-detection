# Pesos dos Modelos (Weights)

Neste diretório encontram-se os pesos oficiais treinados para detecção facial:

## 🏆 `best.pt` (Modelo Otimizado Oficial)
Arquivo com os melhores pesos obtidos no Treinamento 2 (Otimizado - 512px, 922 épocas).

### 📊 Métricas Oficiais Consolidadas (Base: `analise_comparativa/`):
- **mAP @ 0.5:** **65.61%** *(+12.96% | Ganho Relativo: +24.62%)*
- **Recall / Sensibilidade:** **58.84%** *(+11.39% | Ganho Relativo: +24.01%)*
- **Precisão / Precision:** **84.06%** *(+6.24% | Ganho Relativo: +8.01%)*
- **mAP @ 0.5:0.95:** **34.81%** *(+8.16% | Ganho Relativo: +30.59%)*
- **F1-Score Harmônico:** **69.10%** *(+10.29% | Ganho Relativo: +17.49%)*
- **Velocidade de Inferência:** **~3.1 ms** *(~320 FPS em GPU)*

---

### Como utilizar diretamente em Python:

```python
from ultralytics import YOLO

# Carregar o melhor modelo
model = YOLO('models/best.pt')

# Inferência em imagem ou webcam
results = model.predict(source='caminho/para/imagem.jpg', conf=0.25)
```
