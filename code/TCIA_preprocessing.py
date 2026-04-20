import os
import shutil
import numpy as np
import pydicom
import nibabel as nib

def convert_dicom_folder_to_nifti(dicom_folder: str, output_path: str):
    """
    Chuyển đổi 1 thư mục chứa các file DICOM thành 1 file NIfTI (.nii.gz)
    
    Args:
        dicom_folder (str): Đường dẫn đến folder chứa các file .dcm
        output_path (str): Đường dẫn và tên file đầu ra (VD: output/result.nii.gz)
    """
    # 1. Đọc và sắp xếp các file DICOM theo trục Z (ImagePositionPatient)
    dcm_files = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith('.dcm')]
    if not dcm_files:
        print(f"❌ Không tìm thấy file DICOM trong: {dicom_folder}")
        return

    print(f"📁 Đang tải {len(dcm_files)} DICOM slices...")
    slices = [pydicom.dcmread(f) for f in dcm_files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # 2. Xây dựng ma trận ảnh 3D (Volume) và chuyển sang đơn vị Hounsfield (HU)
    volume = np.stack([
        s.pixel_array * float(getattr(s, 'RescaleSlope', 1)) + float(getattr(s, 'RescaleIntercept', 0))
        for s in slices
    ]).astype(np.float32)

    # 3. Tính toán Affine Matrix (Dựa trên thông tin của slice đầu tiên)
    first = slices[0]
    row_spacing, col_spacing = [float(x) for x in getattr(first, 'PixelSpacing', [1.0, 1.0])]
    
    # Tính Slice Thickness
    if len(slices) > 1:
        slice_thickness = abs(float(slices[-1].ImagePositionPatient[2]) - float(slices[0].ImagePositionPatient[2])) / (len(slices) - 1)
    else:
        slice_thickness = float(getattr(first, 'SliceThickness', 1.0))

    # Lấy hướng ảnh (Orientation)
    orientation = getattr(first, 'ImageOrientationPatient', [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    row_dir = np.array(orientation[0:3], dtype=np.float64)
    col_dir = np.array(orientation[3:6], dtype=np.float64)
    slice_dir = np.cross(row_dir, col_dir)

    # Khởi tạo và gán ma trận Affine
    affine = np.eye(4, dtype=np.float64)
    # khung 1: x*y, khung 2: x*z, khung 3: y*z
    affine[0:3, 0] = slice_dir * slice_thickness # z 
    affine[0:3, 1] = col_dir * row_spacing  # y
    affine[0:3, 2] = - row_dir * col_spacing # x
    affine[0:3, 3] = np.array(first.ImagePositionPatient, dtype=np.float64)

    # 4. Lưu thành file NIfTI
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    nifti_img = nib.Nifti1Image(volume, affine=affine)
    nib.save(nifti_img, output_path)
    
    print(f"✅ Đã lưu thành công: {output_path}")
    print(f"📊 Kích thước volume: {volume.shape}")


def find_deepest_dicom_folder(patient_folder: str):
    """Tìm subfolder sâu nhất có chứa file .dcm trong folder bệnh nhân."""
    deepest_folder = None
    deepest_depth = -1
    max_dcm_count = -1

    for root, _, files in os.walk(patient_folder):
        dcm_count = sum(1 for name in files if name.lower().endswith('.dcm'))
        if dcm_count == 0:
            continue

        depth = root.count(os.sep)
        if depth > deepest_depth or (depth == deepest_depth and dcm_count > max_dcm_count):
            deepest_folder = root
            deepest_depth = depth
            max_dcm_count = dcm_count

    return deepest_folder


def get_label_source_path(patient_id: str, label_root: str):
    """Map PANCREAS_00XX -> label00XX.nii.gz trong thư mục label gốc."""
    numeric_id = patient_id.split('_')[-1]
    return os.path.join(label_root, f"label{numeric_id}.nii.gz")

if __name__ == "__main__":
    IMAGES_ROOT = "/home/minhchau/anaconda3/envs/datn/DATN/dataset/TCIA/images_ct_TCIA"
    LABEL_ROOT = "/home/minhchau/anaconda3/envs/datn/DATN/dataset/TCIA/label_TCIA/TCIA_pancreas_labels-02-05-2017"
    OUTPUT_ROOT = "/home/minhchau/anaconda3/envs/datn/DATN/dataset/TCIA_nifti"

    patient_ids = sorted(
        [
            name
            for name in os.listdir(IMAGES_ROOT)
            if name.startswith("PANCREAS_") and os.path.isdir(os.path.join(IMAGES_ROOT, name))
        ]
    )

    for patient_id in patient_ids:
        print(f"\n🩺 Đang xử lý: {patient_id}")
        patient_folder = os.path.join(IMAGES_ROOT, patient_id)

        input_dicom_folder = find_deepest_dicom_folder(patient_folder)
        if input_dicom_folder is None:
            print(f"⚠️ Không tìm thấy thư mục DICOM hợp lệ cho {patient_id}")
            continue

        patient_output_dir = os.path.join(OUTPUT_ROOT, patient_id)
        os.makedirs(patient_output_dir, exist_ok=True)

        # File DICOM lưu tạm ra .nii.gz rồi di chuyển sang đuôi .nifti
        tmp_image_path = os.path.join(patient_output_dir, f"{patient_id}_image.nii.gz")
        final_image_path = os.path.join(patient_output_dir, f"{patient_id}_image.nifti")
        
        convert_dicom_folder_to_nifti(input_dicom_folder, tmp_image_path)
        
        if os.path.exists(tmp_image_path):
            os.replace(tmp_image_path, final_image_path)
            print(f"✅ Đã chuẩn hóa tên ảnh: {final_image_path}")

        # Label: copy `label00XX.nii.gz` sang `PANCREAS_00XX_label.nifti`
        label_src = get_label_source_path(patient_id, LABEL_ROOT)
        final_label_path = os.path.join(patient_output_dir, f"{patient_id}_label.nifti")

        if os.path.exists(label_src):
            shutil.copy2(label_src, final_label_path)
            print(f"✅ Đã copy label: {final_label_path}")
        else:
            print(f"⚠️ Không tìm thấy label cho {patient_id}: {label_src}")

    print(f"\n🏁 Hoàn tất. Output tại: {OUTPUT_ROOT}")