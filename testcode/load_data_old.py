import os
import numpy as np
import cv2
import json
from scipy.spatial.transform import Rotation
from geometry import back_project_to_mesh, image_based_measurement, plane_leastsq
from match_images import match_images_using_features
# Use this script to load images/poses from HL2

class ImagesLoader():

    def __init__(self,data_dir,ref_img_id,manual_boundary=False):
        self.data_dir = data_dir
        self.ref_img_id = ref_img_id
        self.manual_boundary=manual_boundary
        self.load_data()

    def load_data(self):        

        poses_file = os.path.join(self.data_dir,'poses.csv')
        intrinsics_file = os.path.join(self.data_dir,'intrinsics.json')
        mesh_file = os.path.join(self.data_dir,'mesh2.stl')

        with open(intrinsics_file,'r') as f:
            self.intrinsics = json.load(f)
            self.K = np.array(self.intrinsics['camera_matrix'])
            self.dist_coeff = np.array(self.intrinsics['dist_coeff'])
            self.imageSize = (self.intrinsics['width'],self.intrinsics['height'])

        poses = np.loadtxt(poses_file, delimiter=",")
        pose_ref = poses[np.where(poses[:,0]==self.ref_img_id)]
        image_ref = self.load_image(self.ref_img_id,pose_ref)

        # load reference image
        I2 = image_ref[0]
        R2 = image_ref[1]
        C2 = image_ref[2]
        P2 = image_ref[3]
        tvec1 = image_ref[4]

        if self.manual_boundary:
            # click on at least 3 point in the image, then press ENTER
            img_coords,points_3d = image_based_measurement(P2,C2,I2,mesh_file)
        else:
            # or enter Nx2 img coordinates here: np.array([[u1,v1],[u2,v2],...])
            img_coords = np.array([])            
            points_3d = back_project_to_mesh(img_coords,P2,C2,mesh_file)

        n,origin = self.find_plane_normal(C2,points_3d)

        for pose in poses:
            i = pose[0]
            image = self.load_image(i,pose)
            if image is not None:
                I1 = image[0]
                R1 = image[1]
                C1 = image[2]
                P1 = image[3]     
                tvec2 = image[4]

                n1 = R1 @ n
                origin1 = R1 @ origin + tvec1
                d_inv1 = 1.0 / (n1.T @ origin1)

                R = R2 @ R1.T
                t = -R @ tvec1 + tvec2

                H_euclidean = R - (t @ n.T)*d_inv1
                H = self.K @ H_euclidean @ np.linalg.inv(self.K)
                H = H/H[2,2]
                # img_coords_warped = cv2.perspectiveTransform(np.array([img_corners],dtype=np.float32),H)[0]
                I12 = self.warpImage(I1, H)        
                
                # M2 = match_images_using_features(I2,I12)
                # I12 = cv2.warpPerspective(I12,M2,(I12.shape[1],I12.shape[0]))

                im_file2 = os.path.join(self.data_dir,'images_rectified',str(int(i))+'.jpg')
                cv2.imwrite(im_file2,I12)

                cv2.imshow("", I12)                     
                if cv2.waitKey(25) & 0xFF == ord('q'): 
                    pass                 
                
    def warpImage(self,img, H):
        '''warp img2 to img1 with homograph H'''
        w = self.imageSize[0]
        h = self.imageSize[1]        
        pts1 = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)        
        pts2 = cv2.perspectiveTransform(pts1, H)
        pts = np.concatenate((pts1, pts2), axis=0)
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
        t = [-xmin,-ymin]
        Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

        result = cv2.warpPerspective(img, Ht.dot(H), (xmax-xmin, ymax-ymin))
        return result
    

    def load_image(self,i,pose):
            
        pose = pose.reshape(-1)
        C = np.array(pose[1:4]).reshape(-1,1)

        q = pose[4:]
        r = Rotation.from_quat(q)
        Rot = r.as_matrix()
        
        im_file = os.path.join(self.data_dir,'images',str(int(i))+'.jpg')
        
        # Load image (I)
        I = cv2.imread(im_file) 
        if I is None:
            return

        # Construct projection matrix (P)
        R = Rot.T
        t = Rot.T.dot(-C)

        WorldTocameraMatrix = np.hstack([R, t])
        P = self.K @ WorldTocameraMatrix  
        C = C.reshape(-1)

        return [I,R,C,P,t]   

    def find_plane_normal(self,C,points_3d):
        n,p0 = plane_leastsq(points_3d)
        p1 = C - p0
        # if np.dot(p1,n)<0:
        #     n = -1*n        
        return n.reshape(-1,1),p0.reshape(-1,1)

if __name__ == '__main__':
    data_dir = 'poster' # YOUR DATA FOLDER HERE
    ref_img_id = 43
    ImagesLoader(data_dir,ref_img_id,manual_boundary=True)
