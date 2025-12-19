#!/usr/bin/env python3
"""CLI para ler placas de veículos em imagens/vídeos e contar veículos.

Como funciona:
- Detecta regiões retangulares prováveis (contornos com 4 lados) na imagem
- Executa OCR (EasyOCR) nas regiões detectadas
- Normaliza e filtra textos que parecem placas
- Agrega resultados e exporta CSV/relatório

Exemplo de uso:
  python plate_counter.py --input images/ --output results.csv --draw-out out_images/ --min-area 2000

Requisitos: easyocr, opencv-python, imutils, pandas
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import easyocr
from imutils import paths


from plate_utils import normalize_plate


@dataclass
class Detection:
    plate: str
    confidence: float
    image: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h


def detect_plate_candidates(image: np.ndarray, min_area: int = 1000) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, d=11, sigmaColor=75, sigmaSpace=75)
    edged = cv2.Canny(blur, 30, 200)

    # Encontrar contornos
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:50]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            if area > min_area and 2.0 <= (w / float(max(h, 1))) <= 6.0:
                candidates.append((x, y, w, h))
    return candidates


def ocr_plate_from_region(reader: easyocr.Reader, region: np.ndarray, languages: List[str]) -> List[Tuple[str, float]]:
    # easyocr retorna lista de (texto, confidence)
    results = reader.readtext(region, detail=0)
    # detail=0 retorna apenas texto; mas queremos confiança — usar detail=1
    results_with_conf = []
    detailed = reader.readtext(region, detail=1)
    for (bbox, text, conf) in detailed:
        norm = normalize_plate(text)
        if norm:
            results_with_conf.append((norm, float(conf)))
    return results_with_conf


def process_image(path: str, reader: easyocr.Reader, min_area: int = 1000, draw_out: Optional[Path] = None) -> List[Detection]:
    img = cv2.imread(path)
    if img is None:
        return []
    candidates = detect_plate_candidates(img, min_area=min_area)
    detections: List[Detection] = []

    for (x, y, w, h) in candidates:
        crop = img[y : y + h, x : x + w]
        results = ocr_plate_from_region(reader, crop, languages=reader.lang_list)
        for plate, conf in results:
            detections.append(Detection(plate=plate, confidence=conf, image=path, bbox=(x, y, w, h)))

    # Tentar OCR na imagem inteira se nenhuma detecção
    if not detections:
        full = img
        full_results = ocr_plate_from_region(reader, full, languages=reader.lang_list)
        for plate, conf in full_results:
            detections.append(Detection(plate=plate, confidence=conf, image=path, bbox=(0, 0, img.shape[1], img.shape[0])))

    # opcional: desenhar e salvar imagem com caixas
    if draw_out and detections:
        os.makedirs(draw_out, exist_ok=True)
        out_img = img.copy()
        for d in detections:
            x, y, w, h = d.bbox
            cv2.rectangle(out_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(out_img, f"{d.plate} ({d.confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        out_path = Path(draw_out) / Path(path).name
        cv2.imwrite(str(out_path), out_img)

    return detections


def process_images_in_folder(folder: str, reader: easyocr.Reader, min_area: int, draw_out: Optional[str] = None) -> List[Detection]:
    image_paths = list(paths.list_images(folder))
    detections: List[Detection] = []
    for p in image_paths:
        detections.extend(process_image(p, reader, min_area=min_area, draw_out=Path(draw_out) if draw_out else None))
    return detections


def save_csv(detections: Iterable[Detection], out: str) -> None:
    df = pd.DataFrame([
        {"plate": d.plate, "confidence": d.confidence, "image": d.image, "x": d.bbox[0], "y": d.bbox[1], "w": d.bbox[2], "h": d.bbox[3]}
        for d in detections
    ])
    df.to_csv(out, index=False)


def parse_args():
    p = argparse.ArgumentParser(description="Ler placas em imagens/vídeo e contar veículos")
    p.add_argument("--input", required=True, help="Pasta com imagens, arquivo de vídeo ou 'camera'")
    p.add_argument("--output", help="CSV de saída com as detecções")
    p.add_argument("--draw-out", help="Pasta para salvar imagens com caixas desenhadas")
    p.add_argument("--min-area", type=int, default=1500, help="Área mínima (w*h) para considerar candidato")
    p.add_argument("--gpu", action="store_true", help="Habilitar GPU no EasyOCR, se disponível")
    p.add_argument("--lang", default="en", help="Idioma(s) para o EasyOCR (ex: 'en' ou 'pt' ou '['pt','en']')")
    return p.parse_args()


def main():
    args = parse_args()

    langs = [l.strip() for l in args.lang.strip("[]").split(",")] if isinstance(args.lang, str) and "," in args.lang else [args.lang]

    print("Iniciando EasyOCR reader (isso pode demorar na primeira execução)…")
    reader = easyocr.Reader(langs, gpu=args.gpu)

    detections: List[Detection] = []
    if args.input.lower() == "camera":
        print("Leitura da câmera não implementada neste script simples — execute com uma pasta de imagens ou arquivo de vídeo.")
    elif os.path.isdir(args.input):
        detections = process_images_in_folder(args.input, reader, args.min_area, draw_out=args.draw_out)
    elif os.path.isfile(args.input):
        if args.input.lower().endswith(('.mp4', '.avi', '.mov')):
            print("Processamento de vídeo não implementado — use a pasta de frames como entrada.")
        else:
            detections = process_image(args.input, reader, min_area=args.min_area, draw_out=Path(args.draw_out) if args.draw_out else None)
    else:
        raise SystemExit(f"Entrada inválida: {args.input}")

    # Agrupar e contar
    unique_plates = {}
    for d in detections:
        if d.plate not in unique_plates or d.confidence > unique_plates[d.plate].confidence:
            unique_plates[d.plate] = d

    print(f"Imagens processadas: {len(set([d.image for d in detections]))}")
    print(f"Detecções (total): {len(detections)}")
    print(f"Placas únicas: {len(unique_plates)}")

    if args.output:
        save_csv(detections, args.output)
        print(f"Resultados salvos em {args.output}")

    # imprimir resumo por placa
    print("-- Resumo por placa --")
    for plate, d in unique_plates.items():
        print(f"{plate}: confidence={d.confidence:.2f}, image={Path(d.image).name}")


if __name__ == "__main__":
    main()
