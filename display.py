import cv2
import numpy as np

"""
    Customize display setting, including annotation of the output video.
    Direct helper function of vid_pano_track.py.
"""

# TEXT FORMATs:
font=cv2.FONT_HERSHEY_SIMPLEX
fontScale=2
thickness=3
margin=15

# Video scene detection display format.
text_new_scene = "New Scene"
text_size_new_scene = cv2.getTextSize(text_new_scene, font, fontScale, thickness)
text_width_new_scene = text_size_new_scene[0][0]
text_height_new_scene = text_size_new_scene[0][1]

# Face detection number count display format.
text_people_num = "000"
text_size_people_num = cv2.getTextSize(text_people_num, font, fontScale, thickness)
text_width_people_num = text_size_people_num[0][0]
text_height_people_num = text_size_people_num[0][1]


"""
    Return output frame dimension based on pre-defined constraints.
"""
def output_shape(PB, FP, new_width, new_height, maxWidth, maxHeight):
    if PB:
        # Resize dimension for each video frame.
        if (new_height*maxWidth/new_width) <= maxHeight:
            reduce_width = maxWidth
            reduce_height = int(new_height*maxWidth/new_width)
        else:
            reduce_width = int(new_width*maxHeight/new_height)
            reduce_height = maxHeight
        
        frame_width = reduce_width
        frame_height = reduce_height
    else:
        frame_width = new_width
        frame_height = new_height

    if FP:
        frame_width*=2
        frame_height*=2

    return (frame_width, frame_height)


"""
    Return homogrpahy sub-frame based on pre-defined constraints.
"""
def homo_not_match_subframe(PR, canvas, new_width, new_height, new_coord, frame1, frame2):
    # Initialize sub-frame.
    result_img = canvas.copy()
    
    if PR:
        canvas[new_coord[1]:new_coord[1]+frame1.shape[0],
               new_coord[0]:new_coord[0]+frame1.shape[1]] = frame1
    else:
        new_up_bound = int((new_height-frame1.shape[0]-frame2.shape[0])/2.)
        new_left_bound = int((new_width-frame1.shape[1]-frame2.shape[1])/2.)
        result_img[new_up_bound:new_up_bound+frame1.shape[0],
                 new_left_bound:new_left_bound+frame1.shape[1]] = frame1
        result_img[new_up_bound+frame1.shape[0]:new_up_bound+frame1.shape[0]+frame2.shape[0],
                 new_left_bound+frame1.shape[1]:new_left_bound+frame1.shape[1]+frame2.shape[1]] = frame2

    return result_img


"""
    Sub-frame sized coordinates to transform from the panorama to a pre-defined output canvas.
"""
def pano_trans_bounds(pano_coord, new_coord,
                      new_width, new_height,
                      width1, height1,
                      panorama_shape):
    pano_left_bound = pano_coord[2]-new_coord[0]
    pano_right_bound = pano_coord[3]+new_coord[0]
    pano_up_bound = pano_coord[0]-new_coord[1]
    pano_down_bound = pano_coord[1]+new_coord[1]
    
    new_left_bound = 0
    new_right_bound = new_width
    new_up_bound = 0
    new_down_bound = new_height

    if pano_left_bound < 0:
        new_left_bound -= pano_left_bound
            #= new_coord[0]-pano_coord[2]
        pano_left_bound = 0

    if pano_right_bound > panorama_shape[1]:
        new_right_bound -= (pano_right_bound-panorama_shape[1])
            #= new_coord[0]+width1+(panorama_shape[1]-pano_coord[3])
        pano_right_bound = panorama_shape[1]

    if pano_up_bound < 0:
        new_up_bound -= pano_up_bound
            #= new_coord[1]-pano_coord[0]
        pano_up_bound = 0

    if pano_down_bound > panorama_shape[0]:
        new_down_bound -= (pano_down_bound-panorama_shape[0])
            #= new_coord[1]+height1+(panorama_shape[0]-pano_coord[1])
        pano_down_bound = panorama_shape[0]

    return ((pano_left_bound, pano_right_bound, pano_up_bound, pano_down_bound),
            (new_left_bound,  new_right_bound,  new_up_bound,  new_down_bound))


"""
   Format output frame in pre-defined sections.
   FP = [panorama (partial-)result, common region of the two cameras, camera#1, camera#2]
   else = panorama (partial-)result
"""
def format_output_frame(FP, canvas, new_coord, result_img, common_frame, frame1, frame2):
    if FP:
        pro_frame = np.concatenate((result_img, common_frame), axis=1)
        frame1_reshape = canvas.copy()
        frame1_reshape[new_coord[1]:new_coord[1]+frame1.shape[0],
                       new_coord[0]:new_coord[0]+frame1.shape[1]]=frame1
        frame2_reshape = canvas.copy()
        frame2_reshape[new_coord[1]:new_coord[1]+frame2.shape[0],
                       new_coord[0]:new_coord[0]+frame2.shape[1]]=frame2
        org_frame = np.concatenate((frame1_reshape,frame2_reshape),axis=1)

        new_frame = np.uint8(np.concatenate((pro_frame, org_frame), axis=0))
    else:
        new_frame = np.uint8(result_img)

    return new_frame


"""
    Label a recognized face with a rectagle around the head and the name in sub-frames image.
"""
def label_faces(sub_frame, faces, names, names_colors):
    for i, (x,y,w,h) in enumerate(faces):
        cv2.rectangle(sub_frame,
            (int(x),int(y)),(int(x+w),int(y+h)),
            names_colors[names[i]],
            8)
        cv2.putText(sub_frame,
            names[i],
            (int(x),int(y)),
            font,
            fontScale,
            (0,225,0),
            thickness*2)

    return sub_frame


"""
    Display text for new scene.
"""
def display_new_scene_text(new_frame, new_width, new_height):
    
    cv2.rectangle(new_frame,(new_width-text_width_new_scene-margin*2,text_height_people_num+margin*2),
                            (new_width,text_height_people_num+text_height_new_scene+margin*4),(255,255,255),-1)
    cv2.putText(new_frame,
                text_new_scene,
                (new_width-text_width_new_scene-margin, text_height_people_num+margin*6),
                font,
                fontScale,
                (255,0,0),
                thickness)

    return new_frame


"""
    Display text for people number counts for each sub-frame in new scene.
    The following four functions calculate the upper-left, upper-right, lower-left, lower-right
    sub-frames respectively.
"""
def display_ppl_num_text_ul(new_frame, new_width, ppl_num_0, ppl_num_1, ppl_num_2, ppl_num_3):
    
    cv2.rectangle(new_frame,(new_width-text_width_people_num-margin*2,0),
                      (new_width,text_height_people_num+margin*2),(255,255,255),-1)
    cv2.putText(new_frame,
                str(max(0, ppl_num_0, (ppl_num_2+ppl_num_3-ppl_num_1))),
                (new_width-text_width_people_num-margin, margin*4),
                font,
                fontScale,
                (0,0,255),
                thickness)

    return new_frame

def display_ppl_num_text_ur(new_frame, new_width, ppl_num_1):

    cv2.rectangle(new_frame,(new_width*2-text_width_people_num-margin*2,0),
                 (new_width*2,text_height_people_num+margin*2),(255,255,255),-1)
    cv2.putText(new_frame,
                str(ppl_num_1),
                (new_width*2-text_width_people_num-margin, margin*4),
                font,
                fontScale,
                (0,0,255),
                thickness)

    return new_frame

def display_ppl_num_text_bl(new_frame, new_width, new_height, ppl_num_2):
    
    cv2.rectangle(new_frame,(new_width-text_width_people_num-margin*2,new_height),
                (new_width,new_height+text_height_people_num+margin*2),(255,255,255),-1)
    cv2.putText(new_frame,
                str(ppl_num_2),
                (new_width-text_width_people_num-margin, new_height+margin*4),
                font,
                fontScale,
                (0,0,255),
                thickness)
                
    return new_frame

def display_ppl_num_text_br(new_frame, new_width, new_height, ppl_num_3):
    
    cv2.rectangle(new_frame,(new_width*2-text_width_people_num-margin*2,new_height),
                 (new_width*2,new_height+text_height_people_num+margin*2),(255,255,255),-1)
    cv2.putText(new_frame,
                str(ppl_num_3),
                (new_width*2-text_width_people_num-margin, new_height+margin*4),
                font,
                fontScale,
                (0,0,255),
                thickness)
                
    return new_frame
