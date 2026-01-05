# src/utils/logger.py
import logging
import sys
from pathlib import Path

def setup_logger(name="LungCancerApp", log_file="app.log"):
    """
    Központi logger konfiguráció.
    A konzolra és fájlba is ír egyszerre.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Formátum: [Idő] [Szint] Üzenet
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Fájlba írás
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO) # Fájlba csak a fontos infók mennek

    # Konzolra írás
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG) # Konzolra minden mehet fejlesztéskor

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger