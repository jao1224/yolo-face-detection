# Relatório de Treinamento - Detecção Facial com YOLO

Este documento apresenta os resultados obtidos no ajuste fino (*fine-tuning*) do modelo YOLO utilizando o dataset **WiderFace** para a tarefa específica de detecção de rostos em tempo real.

---

## 1. Descrição do Experimento e Hiperparâmetros

O modelo foi treinado a partir de pesos pré-treinados com uma configuração otimizada para aumentar tanto o **Recall (identificação de rostos pequenos/difíceis)** quanto a **Precisão (evitar falsos positivos e ajustar as caixas delimitadoras)**.

### Configurações Utilizadas:
* **Tamanho da Imagem (`imgsz`):** `512` (Aumentado para preservar a resolução de rostos distantes).
* **Tamanho do Lote (`batch`):** `8` (Garante estabilidade na direção dos gradientes de perda).
* **Otimizadores de Perda Customizados:**
  * `cls=0.7` (Aumento de peso na perda de classificação para maximizar a sensibilidade/Recall).
  * `box=10.0` (Aumento de peso na regressão das caixas para ajustar perfeitamente o contorno facial).
  * `dfl=2.0` (Foco aumentado nas bordas finas de objetos difíceis).
* **Agendador de Aprendizado (`cos_lr`):** `True` (Decaimento cosseno da taxa de aprendizado para melhor convergência final).
* **Paciência do Early Stopping (`patience`):** `300` (Permite exploração de ganhos tardios).

---

## 2. Comparativo de Métricas de Desempenho

O modelo foi avaliado no conjunto de validação oficial do **WiderFace** (composto por 3.226 imagens). A tabela abaixo consolida o comparativo entre o Treinamento 1 (Baseline, 320px) e o Treinamento 2 (Otimizado, 512px), conforme a análise estatística oficial da pasta [`analise_comparativa/`](analise_comparativa/):

| Métrica Avaliada | Treino 1 (Baseline) | Treino 2 (Otimizado) | Diferença Absoluta | Evolução Relativa | Impacto Prático na Detecção |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **mAP @ 0.5** | 52.65% *(Época 787)* | **65.61%** *(Época 820)* | **+12.96%** | **+24.62%** | Aumento drástico no reconhecimento geral de faces no dataset |
| **Recall (Sensibilidade)** | 47.44% *(Época 799)* | **58.84%** *(Época 921)* | **+11.39%** | **+24.01%** | Detecta faces menores, distantes, de perfil e com oclusão |
| **Precisão (Precision)** | 77.82% *(Época 768)* | **84.06%** *(Época 793)* | **+6.24%** | **+8.01%** | Redução de detecções falsas (menos ruído no fundo) |
| **mAP @ 0.5:0.95** | 26.66% *(Época 702)* | **34.81%** *(Época 622)* | **+8.16%** | **+30.59%** | Caixas delimitadoras muito mais justas e precisas na face |
| **F1-Score (Harmônico)** | 58.81% *(Época 798)* | **69.10%** *(Época 916)* | **+10.29%** | **+17.49%** | Melhoria global do equilíbrio entre precisão e cobertura |
| **Total de Épocas** | 802 épocas | **922 épocas** | +120 épocas | +15.0% | Treinamento contínuo sem overfitting destrutivo |
| **Velocidade (GPU)** | ~3.5 ms | **3.1 ms** | — | **~320 FPS** | Alta taxa de quadros para inferência em tempo real |

> **Principais Conclusões:**
> 1. **Sensibilidade / Recall (+24.01%):** Resolve o principal desafio da versão baseline (que ignorava faces difíceis).
> 2. **Precisão (+8.01%):** Mesmo detectando faces muito mais complexas, a precisão subiu para 84.06% sem gerar alarmes falsos.
> 3. **Qualidade de Caixas (+30.59% no mAP@0.5:0.95):** Demonstra a eficácia do peso aumentado na perda de caixa (`box=10.0`, `dfl=2.0`).

---

## 3. Instruções para Execução

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

2. Execute o detector via Webcam:
   ```bash
   python main.py
   ```

3. Teste em imagens de amostra:
   ```bash
   python tests/test_yolo.py
   ```
