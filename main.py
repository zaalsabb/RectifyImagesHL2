import os
import numpy as np
import cv2
import json
from scipy.spatial.transform import Rotation
from geometry import back_project_to_mesh, image_based_measurement, ProjectToImage

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
        C2 = image_ref[2]
        P2 = image_ref[3]
        tvec1 = image_ref[4]

        if self.manual_boundary:
            # click on (at least) 4 corners in the image, then press ENTER
            img_coords2,points_3d = image_based_measurement(P2,C2,I2,mesh_file)
        else:
            # or enter Nx2 img coordinates here: np.array([[u1,v1],[u2,v2],...])
            img_coords2 = np.array([])            
            points_3d = back_project_to_mesh(img_coords2,P2,C2,mesh_file)

        for pose in poses:
            i = pose[0]
            image = self.load_image(i,pose)
            if image is not None:
                I1 = image[0]
                P1 = image[3]     
                img_coords1 = ProjectToImage(P1,points_3d)

                H=cv2.findHomography(img_coords1,img_coords2)[0]                
                I12 = cv2.warpPerspective(I1, H, (self.imageSize[0],self.imageSize[1]))

                im_file2 = os.path.join(self.data_dir,'images_rectified',str(int(i))+'.jpg')
                cv2.imwrite(im_file2,I12)

                cv2.imshow("", cv2.resize(I12,(int(I12.shape[1]/2),int(I12.shape[0]/2))))
                if cv2.waitKey(25) & 0xFF == ord('q'): 
                    pass                 


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

if __name__ == '__main__':
    data_dir = 'poster' # YOUR DATA FOLDER HERE
    ref_img_id = 40     # reference image id
    ImagesLoader(data_dir,ref_img_id,manual_boundary=True)
