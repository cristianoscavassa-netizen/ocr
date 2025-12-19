import re
from typing import Optional

PLATE_RE = re.compile(r"[A-Z0-9]{5,8}")


def normalize_plate(text: str) -> Optional[str]:
    """Normaliza texto capturado por OCR e tenta extrair uma placa válida.

    - Remove caracteres não alfanuméricos
    - Transforma em maiúsculas
    - Retorna substring que bate com regex de placa ou None
    """
    if not isinstance(text, str):
        return None
    s = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    if not s:
        return None
    m = PLATE_RE.search(s)
    return m.group(0) if m else None
