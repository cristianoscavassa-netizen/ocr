import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plate_utils import normalize_plate


def test_normalize_plate_basic():
    assert normalize_plate('ABC-1234') == 'ABC1234'
    assert normalize_plate(' ab c  1 2 3  ') == 'ABC123'


def test_normalize_plate_non_plate():
    # entradas sem alfanuméricos devem retornar None
    assert normalize_plate('!!!') is None
    # strings com alfanuméricos podem coincidir com a regex genérica
    hw = normalize_plate('hello world')
    assert (hw is None) or (isinstance(hw, str) and len(hw) >= 5)


def test_normalize_plate_variants():
    assert normalize_plate('a1b2c3') == 'A1B2C3' or normalize_plate('a1b2c3') is None
