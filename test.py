import pytest
import numpy as np
import nibabel as nib
import pydicom
import os
from datetime import datetime
from unittest.mock import patch, MagicMock

# Import the function to be tested
from main import nifti_to_dicom

@pytest.fixture
def mock_nifti_file():
    # Create a mock NIFTI file
    data = np.random.rand(10, 10, 10)
    affine = np.eye(4)
    return nib.Nifti1Image(data, affine)

@pytest.fixture
def mock_output_folder(tmp_path):
    # Create a temporary directory for output
    return tmp_path / "output"

@patch('main.zoom')  # Patch zoom in the main module
@patch('nibabel.load')
@patch('main.generate_uid')  # Patch generate_uid in the main module
def test_nifti_to_dicom(mock_generate_uid, mock_nib_load, mock_zoom, mock_nifti_file, mock_output_folder):
    # Set up mocks
    mock_nib_load.return_value = mock_nifti_file
    mock_zoom.return_value = np.random.rand(512, 512, 970)
    mock_generate_uid.return_value = "1.2.3.4.5.6"

    # Mock the get_fdata method
    mock_nifti_file.get_fdata = MagicMock(return_value=np.random.rand(10, 10, 10))

    # Call the function
    nifti_to_dicom("fake_nifti.nii", str(mock_output_folder))

    # Debug prints
    print(f"mock_nib_load called: {mock_nib_load.called}")
    print(f"mock_zoom called: {mock_zoom.called}")
    print(f"mock_generate_uid called: {mock_generate_uid.called}")

    # Assertions
    assert mock_nib_load.called
    assert mock_zoom.called
    assert mock_generate_uid.called
    assert os.path.exists(mock_output_folder)
    
    # Check if DICOM files were created
    dicom_files = list(mock_output_folder.glob("*.dcm"))
    assert len(dicom_files) == 970  # Expected number of slices

    # Check content of a DICOM file
    if dicom_files:
        ds = pydicom.dcmread(str(dicom_files[0]))
        assert ds.PatientName == "Anonymous"
        assert ds.PatientID == "12345"
        assert ds.Modality == "CT"
        assert ds.Columns == 512
        assert ds.Rows == 512
        assert ds.SamplesPerPixel == 1
        assert ds.PhotometricInterpretation == "MONOCHROME2"
        assert ds.PixelRepresentation == 0
        assert ds.HighBit == 15
        assert ds.BitsStored == 16
        assert ds.BitsAllocated == 16

    print(f"Test complete. {len(dicom_files)} DICOM files created in {mock_output_folder}")

def test_simple():
    assert True

print("Test file is being executed.")