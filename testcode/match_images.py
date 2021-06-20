import numpy as np
import cv2

def match_images_using_features(img1,img2):

    features1,features2 = find_features(img1,img2)

    M2 = cv2.estimateAffinePartial2D(features2, features1, cv2.RANSAC)[0]
    M2 = np.vstack([M2,[0,0,1]])

    return M2


def find_features(img1,img2):
    MIN_MATCH_COUNT = 3

    # Initiate SIFT detector
    detector = cv2.BRISK_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = detector.detectAndCompute(img1,None)
    kp2, des2 = detector.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher()
    # Match descriptors.
    matches = bf.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,2)
            
        return src_pts,dst_pts
    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))        
        return None

    