# LungDxStudio Application
[Magyar nyelvű leírás itt / Hungarian version here](README.hu.md)

LungDxStudio is a desktop application designed for training Machine Learning models using CT scans strictly annotated by medical professionals.

#### Fájl struktúra
```text
LungDxStudio/
├── .github/                # CI/CD (GitHub Actions)
│   └── workflows/          # Testing and build processes
├── Data/                   # Temporary folders for testing (git-ignored)
├── docker/                 # Dockerfile and docker-compose
├── docs/                   # Documentation
├── build/                  # Development build processes
├── dist/                   # Packaged desktop application
├── src/                    # Source code root
│   ├── app.py              # Entry point (Main)
│   ├── config.py           # Configuration management (Class-based instead of config.json)
│   ├── core/               # Business logic (Model and Service layer) 
│   │   ├── data_prep
│   │   │   └── annotation_parser.py    # Parsing and interpreting annotation files (e.g., XML)
│   │   ├── processing
│   │   │   └── tumor_processor.py      # Background thread (QThread) for batch processing of cancerous CT slices
│   │   ├── learning
│   │   │   ├── feature_extractor.py    # Feature extraction and dataset assembly 
│   │   │   └── training_logic.py       # XGBoost model training and evaluation [cite: 31, 33]
│   │   ├── annotation_handler.py # Processing XML-based annotations (LIDC-IDRI format)
│   │   ├── data_manager.py       # Indexing and pairing DICOM images with XML annotations
|   |   ├── lsmc.py               # CT slice loading, HU conversion, and lung mask generation
│   │   └── segmentation/         # Algorithms 
│   │       ├── feature_extractor.py    # Feature extraction, Gabor filters, active contour (snake) algorithm 
│   │       └── lung_segmenter.py       # Lung segmentation on CT scans 
│   ├── gui/                # View layer (Modern GUI) 
│   │   ├── main_window.py  # Display only!
│   │   ├── widgets/        # Custom components (e.g., DICOM viewer widget)
│   │   └── styles/         # Themes, CSS, or QSS files
│   ├── ml/                 # Machine Learning module 
│   │   ├── trainer.py      # XGBoost training logic [cite: 13, 33]
│   │   └── predictor.py    # Prediction logic
│   └── utils/              # Utility functions (logger, math)
├── tests/                  # Unit and Integration tests (pytest)
├── requirements.txt
├── test_data_loading.py
└── README.md
```
#### Architecture: MVVM (Model-View-ViewModel)
- View (GUI): Located in src/gui. This layer never performs calculations. It only handles display and forwards user inputs.
- ViewModel: Connects the GUI to the logic. It contains commands (e.g., start_processing_command) and manages Dask or threading logic to ensure the GUI remains responsive.
- Model (Core): Located in src/core. Contains pure Python classes for DICOM handling and segmentation that are independent of the GUI, ensuring high testability.

#### Data Table (.npz files)
```text
Index	Variable Name	        .npz Key	    Content
0	    origin_img	            original	    Original CT slice (float32) 
1	    segmented_parenchyma	parenchyma	    Lung-masked image (LSMC) 
2	    masked_tumor	        masked_tumor	Tumor only (black background) 
3	    inverted_masked_roi	    inverted_roi	ROI area excluding the tumor
4	    tumor_mask_label	    label	        Tumor classification (e.g., 'A') 
5	    patient_id	            patient_id	    Patient identification
```

#### Install
The application requires Python 3.12.9 and specialized libraries.
```bash
pip install PyQt6 PyQt6-Fluent-Widgets
```

#### CT Imaging Technical Information
- SliceThickness: Represents vertical resolution.
- PixelSpacing: Defines the physical area covered by a single pixel (e.g., 0.7mm x 0.7mm).