# FacialRecognition

FacialRecognition is a Python repo that can detect faces in a video!

## Installation

First, you need to have Python (3.6+) installed and disable PATH limit (preferably).
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install a number of requirements.

```
pip uninstall opencv-python opencv-contrib-python keras
pip install --upgrade opencv-python==3.3.0.10 opencv-contrib-python==3.3.0.10 
pip install --upgrade numpy scikit-image imutils mtcnn tensorflow shutil matplotlib scipy scikit-learn
pip install -I keras==2.1.6
```

## Usage

```
// python vid_pano_track.py [input video 1] [input video 2] [FPS]
python vid_pano_track.py demo_vids/d1-1.mp4 demo_vids/d1-2.mp4 30
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
