"""
Punto de entrada del Simulador de EDE Autónoma.

Uso:
    python main.py
"""

import sys
import os

# Agregar src/ al path para que los imports funcionen
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from gui import main

if __name__ == "__main__":
    main()
