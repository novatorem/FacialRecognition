# reference: https://www.kaggle.com/asymptote/homography-estimate-stitching-two-imag
import numpy as np
import cv2

"""
    Helper functions for homography image stitching.
"""

def warp_images(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    
    list_of_points_1 = np.float32([[0,0], [0,rows1], [cols1, rows1], [cols1,0]]).reshape(-1,1,2)
    temp_points = np.float32([[0,0], [0,rows2], [cols2, rows2], [cols2,0]]).reshape(-1,1,2)
    
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)
    
    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])
    
    trans_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    output_img = trans_img.copy()
    output_img[translation_dist[1]:rows1+translation_dist[1],translation_dist[0]:cols1+translation_dist[0]] = img1
    
    return (output_img,
            trans_img,
            [translation_dist[1],rows1+translation_dist[1],translation_dist[0],cols1+translation_dist[0]])


def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
    # initialize the output visualization image
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB
    
    # loop over the matches
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # only process the match if the keypoint was successfully
        # matched
        if s == 1:
            # draw the match
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

    # return the visualization
    return vis

def detectAndDescribe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(image, None)
    
    # convert the keypoints from KeyPoint objects to NumPy
    # arrays
    kps = np.float32([kp.pt for kp in kps])
    
    # return a tuple of keypoints and features
    return (kps, features)

def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
    # compute the raw matches and initialize the list of actual
    # matches
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []
    
    # loop over the raw matches
    for m in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
        # computing a homography requires at least 4 matches
        if len(matches) > 30:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            
            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojThresh)
                                             
                                             # return the matches along with the homograpy matrix
                                             # and status of each matched point
            return (matches, H, status)

    # otherwise, no homograpy could be computed
    return None


"""
    Find homogrpahy transformation for stationary camears by taking ransac
    for the best matches across frames.
"""
def homo_stationary_cameras(vid1, vid2, ratio, reprojThresh, reprojThresh2):
    k = 10
    sample_num = 5
    
    cap1 = cv2.VideoCapture(vid1)
    cap2 = cv2.VideoCapture(vid2)
    
    sample_count = 0
    frame_count = 0
    all_pts1 = []
    all_pts2 = []
    
    while cap1.isOpened() and cap2.isOpened(): #take minimum length of the two videos
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if frame_count%10==0 and sample_count<=sample_num:
            (kps1, features1) = detectAndDescribe(frame1)
            (kps2, features2) = detectAndDescribe(frame2)
            # Machtes.
            M = matchKeypoints(kps2, kps1, features2, features1, ratio, reprojThresh)
            if M is not None:
                (matches, H, status) = M
                
                matcher = cv2.DescriptorMatcher_create("BruteForce")
                rawMatches = matcher.knnMatch(features1, features2, 2)
                matches = []
                # loop over the raw matches
                for m in rawMatches:
                    # ensure the distance is within a certain ratio of each
                    # other (i.e. Lowe's ratio test)
                    if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                        matches.append((m[0].trainIdx, m[0].queryIdx))
                    # computing a homography requires at least 4 matches
                    if len(matches) > 30:
                        # construct the two sets of points
                        pts1 = np.float32([kps1[i] for (_, i) in matches])
                        pts2 = np.float32([kps2[i] for (i, _) in matches])
                        
                        if all_pts1 == []:
                            all_pts1 = pts1
                            all_pts2 = pts2
                        else:
                            all_pts1 = np.append(all_pts1, pts1, axis=0)
                            all_pts2 = np.append(all_pts2, pts2, axis=0)
                        sample_count+=1
    
        elif sample_count>sample_num:
            break
        frame_count+=1
    
    # Use RANSAC to determine the best matches among all samples.
    (Hs, status) = cv2.findHomography(all_pts1, all_pts2, cv2.RANSAC, reprojThresh2)
    return Hs

