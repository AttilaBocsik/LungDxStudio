# Dockerfile
FROM python:3.11-slim

# Munkakönyvtár beállítása
WORKDIR /app

# Rendszer szintű függőségek telepítése
# (Szükségesek az OpenCV-hez és a PyQt6-hoz Linuxon)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libxkbcommon-x11-0 \
    libdbus-1-3 \
    libxcb-cursor0 \
    && rm -rf /var/lib/apt/lists/*

# Függőségek másolása és telepítése
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Forráskód másolása
COPY . .

# Környezeti változó, hogy a Python lássa a src modult
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Alapértelmezett parancs (ez felülírható, pl. tanítás indítására)
CMD ["python", "src/gui/main_window.py"]