# LungDxStudio applikáció
#### Fájl struktúra
```Plaintext
LungCancerDetection/
├── .github/                # CI/CD (GitHub Actions)
│   └── workflows/          # tesztelés, build folyamatok
├── data/                   # Ideiglenes mappák teszteléshez (git-ből kizárva)
├── docker/                 # Dockerfile és docker-compose
├── docs/                   # Dokumentáció
├── src/                    # A forráskód gyökere
│   ├── app.py              # Belépési pont (Main)
│   ├── config.py           # Konfiguráció kezelés (config.json helyett osztály)
│   ├── core/               # Üzleti logika (a "Model" és "Service" réteg)
│   │   ├── dicom_handler.py    # Robusztus DICOM betöltés
│   │   ├── annotation_handler.py # XML feldolgozás (javított)
│   │   ├── data_manager.py     # A kettő összerendelése (Dicom <-> XML)
│   │   └── segmentation/       # Az algoritmusok
│   │       ├── watershed.py    # (A régi lung_segmentation...py)
│   │       └── active_contour.py # (A régi project_utils snake része)
│   ├── gui/                # A "View" réteg (Modern GUI)
│   │   ├── main_window.py  # Csak a megjelenítés!
│   │   ├── widgets/        # Egyedi komponensek (pl. DICOM nézegető widget)
│   │   └── styles/         # Témák, CSS vagy QSS fájlok
│   ├── ml/                 # Gépi tanulás modul
│   │   ├── trainer.py      # XGBoost tréning logika
│   │   └── predictor.py    # Predikció
│   └── utils/              # Segédfüggvények (logger, matek)
├── tests/                  # Unit és Integration tesztek (pytest)
├── requirements.txt
├── test_data_loading.py
└── README.md
```
#### Architektúra: MVVM (Model-View-ViewModel)
<p><strong>View (GUI):</strong> src/gui. Soha nem végez számítást. Csak megjelenít és gombnyomást továbbít.</p>
<p><strong>ViewModel:</strong> Ez köti össze a GUI-t az logikával. Itt lesznek a parancsok (pl. start_processing_command). Ez kezeli a Dask/Thread logikát, hogy a GUI ne fagyjon le.</p>
<p><strong>Model (Core):</strong> src/core. Itt vannak a tiszta Python osztályok (DICOM kezelés, szegmentálás), amik semmit nem tudnak a GUI-ról. Így tesztelhetővé válnak.</p>

#### Install
```bash
pip install PyQt6 PyQt6-Fluent-Widgets
```