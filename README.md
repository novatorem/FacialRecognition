// set up environment needed
pip3 uninstall opencv-python opencv-contrib-python keras
pip3 install --upgrade opencv-python==3.3.0.10 opencv-contrib-python==3.3.0.10 
pip3 install --upgrade numpy scikit-image imutils mtcnn tensorflow shutil matplotlib scipy scikit-learn
pip3 install -I keras==2.1.6

// how to run our code
// python3 vid_pano_track.py [input video 1] [input video 2] [FPS]
// example below:
python3 vid_pano_track.py demo_vids/d1-1.mp4 demo_vids/d1-2.mp4 30
