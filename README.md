# Foobar

Foobar is a Python library for dealing with word pluralization.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install foobar
```

## Usage

```python
import foobar

foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)

pip3 uninstall opencv-python opencv-contrib-python keras
pip3 install --upgrade opencv-python==3.3.0.10 opencv-contrib-python==3.3.0.10 
pip3 install --upgrade numpy scikit-image imutils mtcnn tensorflow shutil matplotlib scipy scikit-learn
pip3 install -I keras==2.1.6

// how to run our code
// python3 vid_pano_track.py [input video 1] [input video 2] [FPS]
// example below:
python3 vid_pano_track.py demo_vids/d1-1.mp4 demo_vids/d1-2.mp4 30
