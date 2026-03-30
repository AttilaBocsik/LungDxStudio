import os
import pydicom
import pandas as pd  # Opcionális, de szebb a riport vele


def check_dicom_integrity(root_dir):
    report = []
    required_tags = [
        'InstanceNumber',
        'SliceLocation',
        'ImagePositionPatient',
        'RescaleIntercept',
        'RescaleSlope',
        'SliceThickness'
    ]

    print(f"🔍 Vizsgálat indítása: {root_dir}")

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.dcm', '.dicom')):
                file_path = os.path.join(root, file)
                try:
                    ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                    missing = []

                    for tag in required_tags:
                        if not hasattr(ds, tag):
                            missing.append(tag)

                    if missing:
                        report.append({
                            'File': file,
                            'PatientID': getattr(ds, 'PatientID', 'N/A'),
                            'MissingTags': ", ".join(missing),
                            'Path': file_path
                        })
                except Exception as e:
                    report.append({
                        'File': file,
                        'PatientID': 'CORRUPT',
                        'MissingTags': f"READ_ERROR: {str(e)}",
                        'Path': file_path
                    })

    return pd.DataFrame(report)

# Használat:
df_errors = check_dicom_integrity(r"C:\Users\bocsi\Documents\HibasTrain\DICOM")
print(df_errors)
df_errors.to_csv("dicom_error_report.csv", index=False)