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

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def nifti_to_dicom(nifti_file, output_folder, organ_name):
    # NIfTIファイルを読み込む
    nifti = nib.load(nifti_file)
    nifti_array = nifti.get_fdata()

    # 出力フォルダを作成
    ensure_directory(output_folder)

    # NIfTIデータの各スライスをDICOMファイルとして保存
    for i in tqdm(range(nifti_array.shape[2]), desc=f"Converting {organ_name} to DICOM"):
        # DICOMデータセットを作成
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
        ds.ImagePositionPatient = r"0\0\%d" % i
        ds.ImageOrientationPatient = r"1\0\0\0\1\0"
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.HighBit = 15
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.Columns = nifti_array.shape[0]
        ds.Rows = nifti_array.shape[1]
        ds.PixelSpacing = r"1\1"

        dt = datetime.datetime.now()
        ds.ContentDate = dt.strftime('%Y%m%d')
        ds.ContentTime = dt.strftime('%H%M%S.%f')

        # ピクセルデータを設定
        slice_data = nifti_array[:, :, i]
        
        # 0除算を避けるための処理を追加
        slice_min = np.min(slice_data)
        slice_max = np.max(slice_data)
        if slice_max > slice_min:
            slice_data = ((slice_data - slice_min) / (slice_max - slice_min) * 4095).astype(np.uint16)
        else:
            slice_data = np.zeros_like(slice_data, dtype=np.uint16)
        
        ds.PixelData = slice_data.tobytes()

        # DICOMファイルを保存
        ds.save_as(os.path.join(output_folder, f'{organ_name}_slice_{i:04d}.dcm'), write_like_original=False)

    print(f"Conversion complete for {organ_name}. DICOM files saved in {output_folder}")

def combine_organ_data(dicom_folders, output_file):
    combined_data = None
    organ_labels = {}
    current_label = 1

    for folder in tqdm(dicom_folders, desc="Combining organ data"):
        organ_name = os.path.basename(folder).replace("_dicom", "")
        dicom_files = [f for f in os.listdir(folder) if f.endswith('.dcm')]
        dicom_files.sort()

        organ_data = []
        for file in tqdm(dicom_files, desc=f"Processing {organ_name} DICOM files", leave=False):
            ds = pydicom.dcmread(os.path.join(folder, file))
            organ_data.append(ds.pixel_array)

        organ_data = np.stack(organ_data, axis=-1)

        if combined_data is None:
            combined_data = np.zeros_like(organ_data)

        # 各臓器に一意のラベルを割り当てる
        combined_data[organ_data > 0] = current_label
        organ_labels[organ_name] = current_label
        current_label += 1

    # 結合されたデータをNIfTIファイルとして保存
    ensure_directory(os.path.dirname(output_file))
    nifti_img = nib.Nifti1Image(combined_data, np.eye(4))
    nib.save(nifti_img, output_file)

    print(f"Combined voxel data saved as {output_file}")
    print("Organ labels:", organ_labels)

    return combined_data, organ_labels

def create_simulation_model(combined_data, organ_labels, output_file):
    print("Creating simulation model...")
    # ここでシミュレーションモデルを作成するコードを実装
    # 例: 単純な密度マップを作成
    density_map = np.zeros_like(combined_data, dtype=float)
    
    for organ, label in tqdm(organ_labels.items(), desc="Assigning densities to organs"):
        # 仮の密度値
        if organ == "bone":
            density = 1.9  # g/cm^3
        elif organ == "soft_tissue":
            density = 1.0  # g/cm^3
        elif organ == "lung":
            density = 0.3  # g/cm^3
        else:
            density = 1.0  # デフォルト値

        density_map[combined_data == label] = density

    # 密度マップをファイルに保存（例：NPY形式）
    ensure_directory(os.path.dirname(output_file))
    np.save(output_file, density_map)
    print(f"Simulation model (density map) saved as {output_file}")

def visualize_cross_sections(combined_data, organ_labels, output_file):
    print("Visualizing cross-sections of the simulation model...")
    
    # カラーマップの設定
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan']
    color_map = {label: mcolors.to_rgba(colors[i % len(colors)], alpha=0.5) 
                 for i, label in enumerate(organ_labels.values())}
    
    # 背景色（黒）を追加
    color_map[0] = (0, 0, 0, 1)  # 完全に不透明な黒

    # カスタムカラーマップの作成
    cmap = mcolors.ListedColormap([color_map[key] for key in sorted(color_map.keys())])
    
    # 断面の位置を決定（ここでは中央で切っています）
    x_mid = combined_data.shape[0] // 2
    y_mid = combined_data.shape[1] // 2
    z_mid = combined_data.shape[2] // 2
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # XY平面 (Z軸に垂直な断面)
    xy_plane = combined_data[:, :, z_mid]
    ax1.imshow(xy_plane, cmap=cmap, interpolation='nearest')
    ax1.set_title('XY Plane (Z middle)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # YZ平面 (X軸に垂直な断面)
    yz_plane = combined_data[x_mid, :, :]
    ax2.imshow(yz_plane, cmap=cmap, interpolation='nearest')
    ax2.set_title('YZ Plane (X middle)')
    ax2.set_xlabel('Z')
    ax2.set_ylabel('Y')
    
    # ZX平面 (Y軸に垂直な断面)
    zx_plane = combined_data[:, y_mid, :]
    ax3.imshow(zx_plane, cmap=cmap, interpolation='nearest')
    ax3.set_title('ZX Plane (Y middle)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    
    # 凡例の追加
    patches = [plt.Rectangle((0, 0), 1, 1, fc=color_map[label]) for organ, label in organ_labels.items()]
    plt.legend(patches, organ_labels.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Cross-sectional visualization saved as {output_file}")
    plt.close()

# メイン処理
def main():
    # 各臓器のNIfTIファイルをDICOMに変換
    organs = ["bone"]
    roi_dir = "ROI"
    ensure_directory(roi_dir)
    
    for organ in organs:
        nifti_file = os.path.join(roi_dir, f"{organ}.nii")  # .nii ファイルを処理
        if not os.path.exists(nifti_file):
            nifti_file = os.path.join(roi_dir, f"{organ}.nii.gz")  # .nii.gz ファイルも処理
        
        if not os.path.exists(nifti_file):
            print(f"Warning: NIfTI file for {organ} not found. Skipping...")
            continue
        
        output_folder = os.path.join("output", f"{organ}_dicom")
        nifti_to_dicom(nifti_file, output_folder, organ)

    # 各臓器のDICOMデータを結合
    dicom_folders = [os.path.join("output", f"{organ}_dicom") for organ in organs if os.path.exists(os.path.join("output", f"{organ}_dicom"))]
    combined_output = os.path.join("output", "combined_organs.nii.gz")
    combined_data, organ_labels = combine_organ_data(dicom_folders, combined_output)

    # シミュレーションモデルを作成
    simulation_output = os.path.join("output", "simulation_model.npy")
    create_simulation_model(combined_data, organ_labels, simulation_output)

    # モデルの断面を視覚化
    visualization_output = os.path.join("output", "model_cross_sections.png")
    visualize_cross_sections(combined_data, organ_labels, visualization_output)

if __name__ == "__main__":
    main()