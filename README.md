# 🎯 Real-Time Face Detection with Fine-Tuned YOLO

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![YOLO](https://img.shields.io/badge/Ultralytics-YOLO-00FFFF?logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Sistema de detecção de rostos humanos em tempo real de alta precisão e taxa de quadros ultra-rápida (~320 FPS em GPU), treinado e refinado sobre o dataset **WiderFace** utilizando a arquitetura YOLO.

---

## 🌟 Destaques do Projeto

- ⚡ **Ultra Rápido:** Inferência em ~3.1 ms (~320 FPS), ideal para câmeras de segurança, portarias e robótica.
- 🎯 **Alta Sensibilidade:** Modelo otimizado com resolução de `512px` e perdas customizadas (`cls=0.7`, `box=10.0`, `dfl=2.0`), reduzindo drasticamente falhas em rostos pequenos ou com oclusão.
- 📊 **Análise Comparativa Completa:** Benchmarking documentado comparando o modelo Baseline (320px) vs Modelo Otimizado (512px).
- 🚀 **Pronto para Uso:** Código modular, scripts de validação formal, inferência em lote e suporte direto a Webcam.

---

## 📊 Tabela Comparativa de Desempenho (Evidências de Validação)

Avaliação comparativa consolidada sobre o conjunto oficial de validação do **WiderFace** (3.226 imagens):

| Métrica Avaliada | Treino 1 (Baseline) | Treino 2 (Otimizado) | Diferença Absoluta | Ganho Relativo | Impacto Prático na Detecção |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **mAP @ 0.5** | 52.65% *(Época 787)* | **65.61%** *(Época 820)* | **+12.96%** | **+24.62%** | Aumento drástico no reconhecimento geral de faces no dataset |
| **Recall (Sensibilidade)** | 47.44% *(Época 799)* | **58.84%** *(Época 921)* | **+11.39%** | **+24.01%** | Detecta faces menores, distantes, de perfil e com oclusão |
| **Precisão (Precision)** | 77.82% *(Época 768)* | **84.06%** *(Época 793)* | **+6.24%** | **+8.01%** | Redução de detecções falsas (menos ruído no fundo) |
| **mAP @ 0.5:0.95** | 26.66% *(Época 702)* | **34.81%** *(Época 622)* | **+8.16%** | **+30.59%** | Caixas delimitadoras muito mais justas e precisas na face |
| **F1-Score (Harmônico)** | 58.81% *(Época 798)* | **69.10%** *(Época 916)* | **+10.29%** | **+17.49%** | Melhoria global do equilíbrio entre precisão e cobertura |
| **Total de Épocas** | 802 épocas | **922 épocas** | +120 épocas | +15.0% | Treinamento contínuo sem overfitting destrutivo |
| **Velocidade de Inferência** | ~3.5 ms | **3.1 ms** | — | **~320 FPS** | Fluidez máxima em tempo real para Webcam e Câmeras |

### 📈 Evidências Visuais e Gráficos Comparativos
Os gráficos detalhados gerados automaticamente durante os testes estão disponíveis na pasta [`analise_comparativa/`](analise_comparativa/):
- **Gráfico Comparativo de Barras:** [`analise_comparativa/grafico_comparativo_barras.png`](analise_comparativa/grafico_comparativo_barras.png)
- **Evolução Temporal Época a Época:** [`analise_comparativa/grafico_evolucao_metricas.png`](analise_comparativa/grafico_evolucao_metricas.png)
- **Tabela Estruturada CSV:** [`analise_comparativa/tabela_comparativa_metricas.csv`](analise_comparativa/tabela_comparativa_metricas.csv)
- **Dados Estatísticos JSON:** [`analise_comparativa/estatisticas_completas.json`](analise_comparativa/estatisticas_completas.json)

---

## 📁 Estrutura do Repositório

```text
yolo-face-detection/
│
├── main.py                     # Ponto de entrada: detecção em tempo real via Webcam
├── requirements.txt            # Dependências do projeto
├── data.yaml                   # Configuração de caminhos do dataset (template)
├── RELATORIO_TREINAMENTO.md    # Relatório técnico detalhado do experimento
├── README.md                   # Documentação do projeto
├── .gitignore                  # Regras de exclusão para o Git
│
├── models/                     # Pesos treinados prontos para inferência
│   ├── best.pt                 # Modelo final otimizado (512px, WiderFace)
│   └── README.md
│
├── analise_comparativa/        # Estatísticas, dados JSON, tabelas e gráficos
│   ├── grafico_comparativo_barras.png
│   ├── grafico_evolucao_metricas.png
│   ├── tabela_comparativa_metricas.csv
│   ├── estatisticas_completas.json
│   └── RELATORIO_COMPARATIVO.md
│
├── scripts/                    # Utilitários de treino, teste e validação
│   ├── train.py                # Script para fine-tuning customizado
│   ├── validate_model.py       # Validação oficial do modelo (mAP, P, R, F1)
│   ├── gerar_analise_comparativa.py # Gerador automatizado de gráficos e tabelas
│   └── prepare_wider_face.py   # Conversor de anotações do WiderFace para YOLO
│
├── tests/                      # Amostras e testes em lote
│   ├── test_yolo.py            # Execução de predições em imagens locais
│   ├── hotel1.jpg, hotel2.jpg  # Amostras de teste
│   └── test_results/           # Predições visuais salvas
│
└── runs/                       # Logs brutos e curvas de treino
    └── detect/
        ├── train-9/            # Resultados do Baseline (320px)
        └── train-14/           # Resultados do Modelo Otimizado (512px)
```

---

## 🚀 Como Executar

### 1. Clonar o Repositório e Instalar Dependências
```bash
git clone https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git
cd SEU-REPOSITORIO
pip install -r requirements.txt
```

### 2. Detecção Facial em Tempo Real (Webcam)
```bash
python main.py
```
*(Pressione a tecla **`q`** para encerrar).*

### 3. Teste em Amostras de Imagens Estáticas
```bash
python tests/test_yolo.py
```
As imagens anotadas com as caixas e scores de confiança serão salvas em `tests/test_results/predict_test/`.

### 4. Validação Formal no Dataset Completo
```bash
python scripts/validate_model.py
```

### 5. Regerar os Gráficos da Análise Comparativa
```bash
python scripts/gerar_analise_comparativa.py
```

---

## 🛠️ Hiperparâmetros de Treinamento

O modelo foi refinado com as seguintes configurações:
- **Resolução de Entrada (`imgsz`):** `512`
- **Tamanho do Lote (`batch`):** `8`
- **Otimizadores de Perda:**
  - `cls = 0.7` *(Aumento de sensibilidade para rostos pequenos/médios)*
  - `box = 10.0` *(Enquadramento facial mais ajustado)*
  - `dfl = 2.0` *(Maior detalhamento em bordas e contornos)*
- **Agendador de Taxa de Aprendizado:** `cos_lr = True`
- **Paciência (Early Stopping):** `patience = 300`

---

## 📄 Licença

Este projeto é distribuído sob a licença MIT. Consulte o arquivo [LICENSE](LICENSE) para obter mais informações.
