import os
import numpy as np
import nibabel as nib
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import affine_transform, zoom
import glob

def ensure_directory(directory):
    """ディレクトリが存在しない場合は作成する"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def apply_affine_transform(image, affine, reference_affine):
    """アフィン変換を適用して画像を共通の参照フレームに変換する"""
    transform = np.linalg.inv(reference_affine).dot(affine)
    return affine_transform(image, transform[:3, :3], offset=transform[:3, 3], order=1)

def resize_volume(volume, target_shape):
    """ボリュームを指定されたサイズにリサイズする"""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)

def unify_orientation(nifti_img):
    """NIfTIイメージの向きを統一する"""
    return nib.as_closest_canonical(nifti_img)

def create_dicom_dataset(organ_name, reference_affine, i, nifti_array, target_shape):
    """DICOMデータセットを作成する"""
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # 必須のDICOMタグを設定
    ds.PatientName = "Anonymous"
    ds.PatientID = "123456"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.Modality = "CT"
    ds.SeriesDescription = f"{organ_name} Segmentation"
    ds.SeriesNumber = 1
    ds.InstanceNumber = i + 1

    # 位置情報を設定（リサイズ後のサイズに合わせて調整）
    position = reference_affine.dot(np.array([0, 0, i * (nifti_array.shape[2] / target_shape[2]), 1]))[:3]
    ds.ImagePositionPatient = position.tolist()

    # 画像の向きを設定（統一された向きを使用）
    orientation = reference_affine[:3, :2].T.flatten().tolist()
    ds.ImageOrientationPatient = orientation

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.HighBit = 15
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.Columns = target_shape[0]
    ds.Rows = target_shape[1]
    
    # ピクセルスペーシングを設定（リサイズ後のサイズに合わせて調整）
    original_spacing = np.sqrt(np.sum(reference_affine[:3, :2]**2, axis=0))
    pixel_spacing = original_spacing * (nifti_array.shape[:2] / np.array(target_shape[:2]))
    ds.PixelSpacing = pixel_spacing.tolist()
    ds.SliceThickness = np.abs(reference_affine[2, 2]) * (nifti_array.shape[2] / target_shape[2])

    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    ds.ContentTime = dt.strftime('%H%M%S.%f')

    return ds

def process_slice_data(slice_data):
    """スライスデータの正規化と変換"""
    slice_min = np.min(slice_data)
    slice_max = np.max(slice_data)
    if slice_max > slice_min:
        slice_data = ((slice_data - slice_min) / (slice_max - slice_min) * 4095).astype(np.uint16)
    else:
        slice_data = np.zeros_like(slice_data, dtype=np.uint16)
    return slice_data

def nifti_to_dicom(nifti_file, output_folder, flip_x=False, flip_y=False, flip_z=False):
    # Load NIFTI file
    nifti = nib.load(nifti_file)
    nifti_array = nifti.get_fdata()
    
    # Flip the data along the specified axes if needed
    if flip_x:
        nifti_array = np.flip(nifti_array, axis=0)
    if flip_y:
        nifti_array = np.flip(nifti_array, axis=1)
    if flip_z:
        nifti_array = np.flip(nifti_array, axis=2)
    
    # Resample to 512x512x970
    target_shape = (512, 512, 970)
    zoom_factors = np.array(target_shape) / np.array(nifti_array.shape)
    nifti_array_resampled = zoom(nifti_array, zoom_factors, order=1)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Extract patient information (you may need to customize this)
    patient_name = "Anonymous"
    patient_id = "12345"
    
    # Get current time for StudyDate and StudyTime
    current_time = datetime.datetime.now()
    study_date = current_time.strftime('%Y%m%d')
    study_time = current_time.strftime('%H%M%S')

    # Define spacing and orientation (assuming default values, customize as needed)
    pixel_spacing = [1.0, 1.0]
    slice_thickness = 1.0
    slice_spacing = 1.0  # Distance between slices

    for i in tqdm(range(nifti_array_resampled.shape[2]), desc="Converting NIfTI to DICOM"):
        # Create a new DICOM file
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)

        # Add the data elements
        ds.PatientName = patient_name
        ds.PatientID = patient_id
        ds.Modality = "CT"  # Assuming CT, change if necessary
        ds.SeriesInstanceUID = generate_uid()
        ds.StudyInstanceUID = generate_uid()
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

        ds.StudyDate = study_date
        ds.StudyTime = study_time
        ds.AccessionNumber = ''
        ds.InstanceCreationDate = study_date
        ds.InstanceCreationTime = study_time

        # Image specific information
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.HighBit = 15
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.Columns = 512
        ds.Rows = 512
        ds.InstanceNumber = i + 1
        ds.ImagePositionPatient = [0, 0, i * slice_spacing]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = pixel_spacing
        ds.SliceThickness = slice_thickness

        # Set pixel data
        slice_data = nifti_array_resampled[:, :, i]
        ds.PixelData = slice_data.astype(np.uint16).tobytes()

        # Save the DICOM file
        ds.save_as(os.path.join(output_folder, f'slice_{i+1:04d}.dcm'))

    print(f"Conversion complete. DICOM files saved in {output_folder}")

def combine_organ_data(dicom_folder_list, output_folder):
    combined_data = None
    organ_labels = {}
    
    print("Starting combine_organ_data function")
    print(f"DICOM folders: {dicom_folder_list}")
    print(f"Output folder: {output_folder}")

    for label, folder in enumerate(dicom_folder_list, start=1):
        organ = os.path.basename(folder).replace('_dicom', '')
        print(f"Processing organ: {organ}")
        if not os.path.exists(folder):
            print(f"Warning: Folder for {organ} not found. Skipping...")
            continue
        
        dicom_files = sorted(glob.glob(os.path.join(folder, '*.dcm')))
        if not dicom_files:
            print(f"Warning: No DICOM files found for {organ}. Skipping...")
            continue
        
        print(f"Number of DICOM files for {organ}: {len(dicom_files)}")
        
        # Load the first DICOM file to get the image shape
        ds = pydicom.dcmread(dicom_files[0])
        shape = (ds.Rows, ds.Columns, len(dicom_files))
        
        print(f"Shape for {organ}: {shape}")
        
        # Initialize or update the combined data array
        if combined_data is None:
            combined_data = np.zeros(shape, dtype=np.float32)
        
        # Load and combine DICOM data
        for i, file in enumerate(tqdm(dicom_files, desc=f"Processing {organ}")):
            ds = pydicom.dcmread(file)
            pixel_array = ds.pixel_array.astype(np.float32)
            combined_data[:,:,i] = np.where(pixel_array > 0, label, combined_data[:,:,i])
        
        # Assign a unique label for each organ
        organ_labels[organ] = label
    
    print("Organ labels:", organ_labels)
    
    if combined_data is None:
        print("Error: No data was combined. All folders might be empty or non-existent.")
        return None, None
    
    print(f"Final combined data shape: {combined_data.shape}")
    
    # Create NIfTI image
    nifti_img = nib.Nifti1Image(combined_data, np.eye(4))
    
    # Save NIfTI file
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, 'combined_organs.nii.gz')
    nib.save(nifti_img, output_file)
    
    print(f"Combined NIfTI file saved to: {output_file}")
    
    return combined_data, organ_labels

def create_simulation_model(combined_data, organ_labels, output_file):
    """シミュレーションモデルを作成する"""
    print("Creating simulation model...")
    density_map = np.zeros_like(combined_data, dtype=float)
    
    for organ, label in tqdm(organ_labels.items(), desc="Assigning densities to organs"):
        if organ == "bone":
            density = 1.9  # g/cm^3
        elif organ == "soft_tissue":
            density = 1.0  # g/cm^3
        elif organ == "lung":
            density = 0.3  # g/cm^3
        else:
            density = 1.0  # デフォルト値

        density_map[combined_data == label] = density

    ensure_directory(os.path.dirname(output_file))
    np.save(output_file, density_map)
    print(f"Simulation model (density map) saved as {output_file}")
    
def visualize_dicom(dicom_folder, organ_name):
    """指定されたフォルダ内のDICOMファイルを表示し、XY、YZ、ZX平面の断面を表示する"""
    dicom_files = sorted([f for f in os.listdir(dicom_folder) if f.endswith('.dcm')])
    
    if not dicom_files:
        print(f"No DICOM files found in {dicom_folder}")
        return
    
    try:
        # すべてのDICOMファイルを読み込み、3D配列を作成
        volume = []
        for file in tqdm(dicom_files, desc=f"Loading {organ_name} DICOM files"):
            ds = pydicom.dcmread(os.path.join(dicom_folder, file))
            volume.append(ds.pixel_array)
        volume = np.array(volume)
        
        # 中央のスライスを選択
        x_mid, y_mid, z_mid = [s // 2 for s in volume.shape]
        
        # 3つの平面の断面を取得
        xy_plane = volume[z_mid]
        yz_plane = volume[:, x_mid, :]
        zx_plane = volume[:, :, y_mid].T
        
        # 画像を表示
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        ax1.imshow(xy_plane, cmap='gray')
        ax1.set_title(f"{organ_name} - XY Plane (Z middle)")
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.axis('on')
        
        ax2.imshow(yz_plane, cmap='gray')
        ax2.set_title(f"{organ_name} - YZ Plane (X middle)")
        ax2.set_xlabel('Z')
        ax2.set_ylabel('Y')
        ax2.axis('on')
        
        ax3.imshow(zx_plane, cmap='gray')
        ax3.set_title(f"{organ_name} - ZX Plane (Y middle)")
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        ax3.axis('on')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error processing DICOM files: {e}")

def visualize_cross_sections(combined_data, organ_labels, output_file):
    """シミュレーションモデルの断面を視覚化する"""
    print("Visualizing cross-sections of the simulation model...")
    
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan']
    color_map = {label: mcolors.to_rgba(colors[i % len(colors)], alpha=0.5) 
                 for i, (organ, label) in enumerate(organ_labels.items())}
    color_map[0] = (0, 0, 0, 1)  # 完全に不透明な黒

    cmap = mcolors.ListedColormap([color_map[key] for key in sorted(color_map.keys())])
    
    x_mid = combined_data.shape[0] // 2
    y_mid = combined_data.shape[1] // 2
    z_mid = combined_data.shape[2] // 2
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    xy_plane = combined_data[:, :, z_mid]
    ax1.imshow(xy_plane, cmap=cmap, interpolation='nearest')
    ax1.set_title('XY Plane (Z middle)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    yz_plane = combined_data[x_mid, :, :]
    ax2.imshow(yz_plane, cmap=cmap, interpolation='nearest')
    ax2.set_title('YZ Plane (X middle)')
    ax2.set_xlabel('Z')
    ax2.set_ylabel('Y')
    
    zx_plane = combined_data[:, y_mid, :].T
    ax3.imshow(zx_plane, cmap=cmap, interpolation='nearest')
    ax3.set_title('ZX Plane (Y middle)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    
    patches = [plt.Rectangle((0, 0), 1, 1, fc=color_map[label]) for organ, label in organ_labels.items()]
    plt.legend(patches, organ_labels.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Cross-sectional visualization saved as {output_file}")
    plt.close()

def main():
    print("Starting main function")
    
    # NIfTIファイルの入力フォルダと出力フォルダを指定
    input_folder = 'roi'
    output_folder = 'output'
    combined_output = os.path.join(output_folder, 'combined_organs.nii.gz')

    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Combined output: {combined_output}")

    # 各臓器のNIfTIファイルを処理
    organs = ['bone', 'lung']
    flip_config = {
        'bone': {'flip_x': True, 'flip_y': True, 'flip_z': True},
        'lung': {'flip_x': False, 'flip_y': False, 'flip_z': False}
    }
    dicom_folders = []

    print("Processing organs:")
    for organ in organs:
        nifti_file = os.path.join(input_folder, f'{organ}.nii')
        if os.path.exists(nifti_file):
            print(f"  Found NIfTI file for {organ}")
            dicom_folder = os.path.join(output_folder, f'{organ}_dicom')
            dicom_folders.append(dicom_folder)
            
            # 必要に応じて軸を反転
            flip_x = flip_config[organ]['flip_x']
            flip_y = flip_config[organ]['flip_y']
            flip_z = flip_config[organ]['flip_z']
            nifti_to_dicom(nifti_file, dicom_folder, flip_x=flip_x, flip_y=flip_y, flip_z=flip_z)
            
            # 生成されたDICOMデータを表示
            visualize_dicom(dicom_folder, organ)
        else:
            print(f"  Warning: NIfTI file for {organ} not found. Skipping...")

    print(f"DICOM folders: {dicom_folders}")
    print(f"Combined output folder: {combined_output}")

    # 各臓器のDICOMデータを結合
    combined_data, organ_labels = combine_organ_data(dicom_folders, combined_output)
    
    if combined_data is None or organ_labels is None:
        print("Error: Failed to combine organ data. Exiting...")
        return
    
    print("Combined data shape:", combined_data.shape)
    print("Organ labels:", organ_labels)

    # シミュレーションモデルを作成
    simulation_output = os.path.join("output", "simulation_model.npy")
    create_simulation_model(combined_data, organ_labels, simulation_output)

    # モデルの断面を視覚化
    visualization_output = os.path.join("output", "model_cross_sections.png")
    visualize_cross_sections(combined_data, organ_labels, visualization_output)

if __name__ == "__main__":
    main()
