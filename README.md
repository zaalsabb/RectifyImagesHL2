# Rectify Images using HoloLens 2

Rectify images collected from Microsoft's HoloLens 2 with the help of the spatial mesh.

## Installation

Clone the repository using git:

```bash
git clone https://github.com/zaalsabb/RectifyImagesHL2.git
```
Install requirements using pip:

```bash
cd RectifyImagesHL2
pip install requirements.txt
```

## Usage

1. Select data folder (data_dir) where images/poses/mesh files are stored. (Output rectified images are stored in /images_rectified)

2. Select reference image id (ref_img_id), so that all other images are rectified to this reference image.

3. ```bash
   cd RectifyImagesHL2
   python main.py
   ```

4. Select 2d image points that are back-projected to the spatial mesh by clicking on the screen, or automatically in code (img_coords2). These 3D points then are then projected on to each image, then cv2.findHomography is used to find the Homography transformation.
```python
if __name__ == '__main__':
    data_dir = 'poster' # YOUR DATA FOLDER HERE
    ref_img_id = 40     # reference image id
    ImagesLoader(data_dir,ref_img_id,manual_boundary=True)

```