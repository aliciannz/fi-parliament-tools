import pytest
import fi_parliament_tools.face_extraction as face_extraction 
from typing import List

# ---------------------------
# Test function modify_coords
# ---------------------------
@pytest.mark.parametrize("input_coords, expected_coords", [
    ([5, 5, 10, 10], [5, 5, 5, 5]), 
    ([5, 5, 15, 15], [10, 10, 5, 5])
])
def test_compute_coords(input_coords: List[int], expected_coords: List[int]) -> None:
    """Test function compute_coords with valid inputs"""
    coords = face_extraction.modify_coords(input_coords)
    assert coords == expected_coords

@pytest.mark.parametrize("input_coords", [
    [5, 5, 5, 10],
    [5, 5, 10, 5],
    [10, 5, 5, 15],
])
def test_invalid_coords(input_coords: List[str]) -> None:
    """Test that face crops with 0 length or width are caught."""
    with pytest.raises(ValueError):
        face_extraction.modify_coords(input_coords)

# ---------------------------
# Test function find_nearest_face
# ---------------------------
@pytest.mark.parametrize("input_coords, expected_coords", [
    ([[10, 10, 960, 540], [10, 10, 0, 0],  [15, 15, 900, 500]], 0)
])
def test_find_nearest_face(input_coords: List[int], expected_coords: List[int]) -> None:
    """Test function find_nearest_face with valid inputs"""
    coords = face_extraction.find_nearest_face(input_coords)
    assert coords == expected_coords
    
# ---------------------------
# Test function convert_to_timestamp
# ---------------------------
@pytest.mark.parametrize("input_frame, expected_timestamp", [
    (45, "01.800"),
    (100, "04.000")

])
def test_convert_to_timestamp(input_frame: int, expected_timestamp: str) -> None:
    timestamp = face_extraction.convert_to_timestamp(input_frame)
    assert timestamp == expected_timestamp

