# LungDxStudio applikáció
#### Fájl struktúra
```Plaintext
LungCancerDetection/
├── .github/                # CI/CD (GitHub Actions)
│   └── workflows/          # tesztelés, build folyamatok
├── Data/                   # Ideiglenes mappák teszteléshez (git-ből kizárva)
├── docker/                 # Dockerfile és docker-compose
├── docs/                   # Dokumentáció
├── build/                  # Fejlesztési build folyamatok
├── dist/                   # Ekészített asztali alkalmazás
├── src/                    # A forráskód gyökere
│   ├── app.py              # Belépési pont (Main)
│   ├── config.py           # Konfiguráció kezelés (config.json helyett osztály)
│   ├── core/               # Üzleti logika (a "Model" és "Service" réteg)
│   │   ├── data_prep
│   │   │   └── annotation_parser.py      # Annotációs fájlok (pl. XML) beolvasása és értelmezése
│   │   ├── processing
│   │   │   └── tumor_processor.py      # Háttérszál (QThread) a daganatos CT szeletek kötegelt feldolgozása
│   │   ├── learning
│   │   │   ├── feature_extractor.py    # Képjellemzők kinyerése és adathalmaz összeállítása
│   │   │   └── training_logic.py       # XGBoost modell tanítása és kiértékelése
│   │   ├── annotation_handler.py # XML feldolgozás (javított)
│   │   ├── data_manager.py     # A kettő összerendelése (Dicom <-> XML)
|   |   ├── lsmc.py     # CT szeletek betöltése, HU konverzió és a tüdőmaszk generálása
│   │   └── segmentation/       # Az algoritmusok
│   │       ├── feature_extractor.py    # Képjellemzők kinyerésé, Gabor-szűrők, aktív kontúr (snake) algoritmus
│   │       └── lung_segmenter.py       # CT felvételeken végzett tüdőszegmentálás
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

#### Fájlok
```Plaintext
Eredeti lista indexe	Változónév nálad	.npz fájl kulcsa (Key)	Tartalom
0	                    origin_img	        original	            Az eredeti CT szelet (float32)
1	                    segmented_parenchyma	parenchyma	        A tüdőmaszkolt kép (LSMC)
2	                    masked_tumor	    masked_tumor	        Csak a daganat (körülötte fekete)
3	                    inverted_masked_roi	inverted_roi	        A ROI területe a daganat nélkül
4	                    tumor_mask_label	label	                A daganat típusa (pl. 'A')
5	                    patient_id	        patient_id	            A beteg azonosítója
```

#### Install
```bash
pip install PyQt6 PyQt6-Fluent-Widgets
```

### CT információk
- SliceThickness: Megmutatja a vertikális felbontást.
- PixelSpacing: Megmutatja, mekkora területet fed le egyetlen pixel a valóságban (pl. 0.7mm x 0.7mm).