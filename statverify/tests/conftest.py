"""Делает вспомогательные модули тестов (corpus, corrupt, synth) импортируемыми
по короткому имени, не превращая каталог тестов в пакет."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
