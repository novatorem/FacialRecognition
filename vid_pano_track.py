# ENVIRONMEMNT and PACKAGE
# pip3 install --upgrade opencv-python==3.3.0.10 opencv-contrib-python==3.3.0.10 keras==2.1.6 tensorflow scikit-image imutils mtcnn numpy
# May require to uninstall before install those packages.


import sys
import os.path
import cv2
import numpy as np
from random import randint
import recognization as rec
import recog_setup as rs

# display interface
import display
# panorama transformation
from panorama import warp_images, drawMatches, detectAndDescribe, matchKeypoints, homo_stationary_cameras
# poisson blending
from poisson import poisson_blending
    ### Replaced by recongition: >>>
    # face training and detection
    # from detection import mtcnn_detect
    ### <<<
# new scene recognize
from scene_recognize import mse_maxpool, identify_scene
# face tracking
from tracking import createTrackerByName


"""
    Face detection, recognization and tracking for video panorama.
    python3 vid_pano_track.py vid_name1 vid_name2 fps
    e.x. python3 vid_pano_track.py demo_vids/d1-1.mp4 demo_vids/d1-2.mp4 30
"""
# THRESHOLDs
PR=True #PRIORITY, display frame1 only when the two videos have no agreement.
        # suggestion: True
PB=False #POISSON BLENDING, requires the output video to have a lower resolution.
         # suggestion: False
maxWidth=300
maxHeight=200
FP=True #FOUR PANELs.
SC=True #STATIONARY CAMERAs.
trackerType = "BOOSTING" #Tracking methods:
                         #trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW']

# homography transformation thresholds
ratio=0.7
reprojThresh=5.0
reprojThresh2=20.0

perFrame=2
outputSizeTimes=1 # Output sub-frame dimension = (int(height1 + height2*outputSizeTimes),
                  #                               int(width1 + width2*outputSizeTimes))
                  # So that frame1 is included in full in a sub-frame.
                  # suggest: >=0.8
cutThreshold=10 #suggest: ~10
aroundCutThreshold=7 #suggest: ~7

# new scene threshold
cutTextDuration=5 #suggest: >=5

# recognition threshold
align_method = "mtcnn" # align_method=["mtcnn", "haar"]
ml_method = "svm" # ml_method=["svm", "knn"]

# background colour of output video
canvas_color=(255,255,255)

# output file name, avoid duplicate names
output_file_name = "output.mp4" #sys.argv[1].split("-")[0]+"-output.mp4"
while os.path.exists(output_file_name):
    output_file_name = output_file_name.split(".")[0]+"1.mp4"
print("Result will be saved at /",output_file_name)


"""
    Analyze and precess by frame, and produce video.
"""
def vid_main(vid1, vid2, fps):
    # Open video file for read.
    cap1 = cv2.VideoCapture(vid1)
    cap2 = cv2.VideoCapture(vid2)
    
    # Perpare dataset and recognition models
    FRmodel, ml_model, encoder = rs.set_up(align_method, ml_method)
    print("Successfully set up recongnition system.")
    
    # Determine homography transfroamtion equal for all scene if cameras are stable.
    if SC: H = homo_stationary_cameras(vid2, vid1, ratio, reprojThresh, reprojThresh2)
    
    # Initialize.
    count = 0
    new_scene = False
    new_scene_count = 0
    ppl_counts = [0, 0, 0, 0] # number of people prediction in each sub-frame
    names_0, names_1, names_2, names_3 = ([], [], [], []) # list of name prediction in each sub-frame
    object_trackers = []
    all_names = []
    names_colors = {}  # each person with a pre-collected dataset and is detected in the frame is marked by an unique colour.
    sub_frames = [] # len(sub_frames)=1 if FP==True, else=4
    
    while cap1.isOpened() and cap2.isOpened(): #take minimum length of the two videos
        print(count) # Current frame number
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if ret1 and ret2:
            # At first frame, obtain input frame dimension and create VideoWriter object.
            if count==0:
                height1, width1, layers1 = frame1.shape
                height2, width2, layers2 = frame2.shape
                new_height = int(height1 + height2*outputSizeTimes)
                new_width = int(width1 + width2*outputSizeTimes)
                # Coordinate up-left coordinate so that frame1 with (height1, width1)
                # will be placed at the centre of (new_height, new_width).
                new_coord = [int((new_width-width1)/2.), int((new_height-height1)/2.)]

                # Initialize the dimension of output video format.
                frame_width, frame_height = display.output_shape(PB, FP, new_width, new_height, maxWidth, maxHeight)
                out = cv2.VideoWriter(output_file_name,
                                      cv2.VideoWriter_fourcc(*'MP4V'),
                                      fps,
                                      (frame_width,frame_height))
                
                # Initialize sub-frames.
                mask_img = np.zeros((new_height,new_width)).astype(np.uint8)
                mask_img[new_coord[1]:new_coord[1]+height1,
                         new_coord[0]:new_coord[0]+width1].fill(255)
                canvas = np.zeros((new_height,new_width,3)).astype(np.uint8) # dimension of sub-frame
                if canvas_color != (0,0,0):
                    canvas[:,:,:] = canvas_color


            # SCENE DETECTION:
            # Check video scene change, assuming the two cameras cut at the same time.
            # Else the sorresponding frames do not match.
                pre_frame1 = frame1 #count == 0
            new_frame1_norm_gray = cv2.cvtColor((frame1 * 255.0/frame1.max()).astype('uint8'), cv2.COLOR_BGR2GRAY)
            pre_frame1_norm_gray = cv2.cvtColor((pre_frame1 * 255.0/pre_frame1.max()).astype('uint8'), cv2.COLOR_BGR2GRAY)
            curr_err = mse_maxpool(new_frame1_norm_gray, pre_frame1_norm_gray)
            if count == 0:
                pre_err = curr_err
            curr_err_diff = abs(pre_err - curr_err)
            if count == 0:
                pre_err_diff = curr_err_diff
            new_scene = identify_scene(pre_err_diff, curr_err_diff, pre_err, curr_err, cutThreshold, aroundCutThreshold)
            # Update for the next *necognition* process.
            if new_scene or count==0:
                new_scene_count = cutTextDuration
                object_trackers = []
                all_names = []
                ppl_counts = [0, 0, 0, 0]
#            pre_frame1 = frame1###


            # PANORAMA FEATURE DETECTION:
            # Use the last calculated frame's homography transformation.
            if count%perFrame==0:
                # Create PANORAMA of the two frame.
                # Create describer.
                (kps1, features1) = detectAndDescribe(frame1)
                (kps2, features2) = detectAndDescribe(frame2)
                # Machtes.
                M = matchKeypoints(kps2, kps1, features2, features1, ratio, reprojThresh)

            # If the match is None, then there aren't enough matched keypoints to create a panorama
            # Assume the views are not overlapped, display both videos by simply paste them together as default.
            if M is None and not SC:
                result_img = display.homo_not_match_subframe(PR, canvas, new_width, new_height, new_coord, frame1, frame2)
                common_frame = canvas.copy()
                common_frame[:,:,:] = (0,0,0)
            else:
                if not SC: # Evaluate individual frame's homography transformation when the cameras are not stationary.
                    (matches, H, status) = M
                # Create panorama.
                (panorama, frame2_trans, pano_coord) = warp_images(frame1, frame2, H)

                # Coordinate for transfering panorama coordinates result to a proper sized canvas,
                # with frame1 at the centre.
                pano_bounds, new_bounds = display.pano_trans_bounds(pano_coord, new_coord, new_width, new_height, width1, height1, panorama.shape)
                pano_left_bound, pano_right_bound, pano_up_bound, pano_down_bound = pano_bounds
                new_left_bound,  new_right_bound,  new_up_bound,  new_down_bound  = new_bounds
                
                # frame2 after homogrpahy transformation, for poisson blending and determine the common region.
                target_img = canvas.copy()
                target_img[new_up_bound:new_down_bound, new_left_bound:new_right_bound, :] =\
                    frame2_trans[pano_up_bound:pano_down_bound, pano_left_bound:pano_right_bound, :]


                # POISSON BLENDING:
                if PB:
                    # Input images for poisson blending.
                    source_img = canvas.copy()
                    source_img[new_coord[1]:new_coord[1]+frame1.shape[0],
                               new_coord[0]:new_coord[0]+frame1.shape[1], :] = frame1
                    # Resize the images.
                    source_img = cv2.resize(source_img, (reduce_width, reduce_height))
                    target_img = cv2.resize(target_img, (reduce_width, reduce_height))
                    mask_img = cv2.resize(mask_img, (reduce_width, reduce_height))
                    # Apply Poisson blending.
                    result_img = poisson_blending(source_img, target_img, mask_img)
                else:
                    result_img = canvas.copy()
                    result_img[new_up_bound:new_down_bound, new_left_bound:new_right_bound, :] =\
                        panorama[pano_up_bound:pano_down_bound, pano_left_bound:pano_right_bound, :]

                # Mask indicate the common area of the two frames.
                common_mask = canvas.copy()
                common_mask[:,:,:] = (0,0,0)
                common_mask[target_img==0] = 255
                common_mask[mask_img==0] = 255
                # Apply to output frame.
                common_frame = result_img.copy()
                common_frame[common_mask==255] = 0

            # Find and label faces in each sub-frame.
            if FP:
                sub_frames = [panorama, common_frame, frame1, frame2]
#                cv2.imwrite("sub1.png", panorama) ## Save INDIVIDUAL FRAMES for test purpose
#                cv2.imwrite("sub2.png", common_frame)
#                cv2.imwrite("sub3.png", frame1)
#                cv2.imwrite("sub4.png", frame2)
            else:
                sub_frames = [panorama]

            for i,sub_img in enumerate(sub_frames):
                # Re-perform detection and recognation when a video scene change is detected,
                # then identify people by tracking.
                if new_scene or count==0:
        ### Replaced by recognition: >>>
        #            # FACE DETECTION:
        #            faces, eyes = mtcnn_detect(new_frame)
        ### <<<
                    # FACE RECOGNITION with DETECTION
                    names, patches, faces = rec.predict(sub_img, ml_model, encoder, align_method, FRmodel)
                    ppl_counts[i] = len(names)
                    for name in names:
                        if name not in names_colors:
                            names_colors[name] = (randint(0, 255), randint(0, 255), randint(0, 255))

                    # OBJECT TRACKING:
                    # Create MultiTracker for tracking multiple objects.
                    curr_multiTracker = cv2.MultiTracker_create()
                    for (x,y,w,h) in faces:
                        curr_multiTracker.add(createTrackerByName(trackerType), sub_img, (x,y,w,h))
                    object_trackers.append(curr_multiTracker)
                    all_names.append(names)
                    new_scene = False
                else: # not new scene: previous detected scene = current frame scene
                    # Update face tracking using previouly initialized multiple object tracker.
                    success, faces = object_trackers[i].update(sub_img)

                display.label_faces(sub_img, faces, all_names[i], names_colors)
                # Panorama /main sub-frame
                if i==0:
                    result_img[new_up_bound:new_down_bound, new_left_bound:new_right_bound, :] =\
                    sub_img[pano_up_bound:pano_down_bound, pano_left_bound:pano_right_bound, :] #update labels
#                    cv2.imwrite("sub5.png", result_img) ## Save LABELED MAIN FRAME for test purpose

            # Define new frame.
            new_frame = display.format_output_frame(FP, canvas, new_coord, result_img, common_frame, frame1, frame2)

            # Add text anotations:
            # Display the "New scene" sign in new_scene_count frame, shortly after new cut detection.
            if new_scene_count >= 0:
                new_frame = display.display_new_scene_text(new_frame, new_width, new_height)
            # Display text for number of people.
            new_frame = display.display_ppl_num_text_ul(new_frame, new_width, ppl_counts[0], ppl_counts[1], ppl_counts[2], ppl_counts[3]) # Final count: up-left image.
            if FP:
                new_frame = display.display_ppl_num_text_ur(new_frame, new_width, ppl_counts[1]) # up-right image.
                new_frame = display.display_ppl_num_text_bl(new_frame, new_width, new_height, ppl_counts[2]) # bottom-left image.
                new_frame = display.display_ppl_num_text_br(new_frame, new_width, new_height, ppl_counts[3]) # bottom-right image.
#            if count==0:
#                cv2.imwrite("test.png", new_frame) ## Save FIRST FRAME of output video for test purpose.

            # Write to output video.
            out.write(new_frame)
            
            # Update for the next *frame*.
            pre_frame1_norm_gray = new_frame1_norm_gray.copy()
            pre_err = curr_err
            pre_err_diff = curr_err_diff
            new_scene_count -= 1
            names_0, names_1, names_2, names_3 = ([], [], [], [])
            sub_frames = []
            count += 1
        
        elif cv2.waitKey(1) & 0xFF == ord('q'): # quit on ESC button
            break
        else:
            break

    # Release everything if job is finished
    cap1.release()
    cap2.release()
    print("Video analysis complete!")
    out.release()
    cv2.destroyAllWindows() # destroy all the opened windows
    return


if __name__ == '__main__':
    vid_main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
