# OCR Plate Counter 🔧

Pequeno CLI em Python para **ler placas de veículos** em imagens e **contar veículos** únicos.

## Requisitos

- Python 3.9+
- pip install -r requirements.txt

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Observação: EasyOCR baixa modelos na primeira execução. Para usar GPU, instale dependências CUDA e execute com `--gpu`.

## Uso

```bash
# Processar pasta de imagens
python plate_counter.py --input images/ --output results.csv --draw-out out_images/ --min-area 1500

# Processar uma imagem única
python plate_counter.py --input samples/car1.jpg --output detections.csv
```

Parâmetros principais:
- `--input`: pasta com imagens (ou arquivo de imagem)
- `--output`: arquivo CSV para salvar detecções
- `--draw-out`: pasta para salvar imagens anotadas
- `--min-area`: ajustar sensibilidade de detecção
- `--lang`: idioma do OCR (ex: `pt` ou `en`)
- `--gpu`: usar GPU (se disponível)

## Limitações
- Método de detecção de placas é heurístico (contornos + proporção). Para melhor robustez, recomendo usar um detector treinado (YOLO/DeepLearning) para localizar placas antes do OCR.

## Contribuições
Sinta-se à vontade para abrir PRs com melhorias (suporte a vídeo, detector baseado em DL, tracking para contagem em fluxo, etc.).
