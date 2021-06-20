                # cv2.imshow('',image[0])
                # if cv2.waitKey(25) & 0xFF == ord('q'): 
                #     pass 
                I1 = image_ref[0]
                R1 = image_ref[1]
                t1 = image_ref[2]
                P1 = image_ref[3]
                
                I2 = image[0]
                R2 = image[1]
                t2 = image[2]
                P2 = image[3]     

                F=vgg_F_from_P(P1, P2)          

                R = R1.T @ R2
                t = R1.T @ (t2 - t1)

                rectify_scale = 1 # 0=full crop, 1=no crop
                R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.K, self.dist_coeff, self.K, self.dist_coeff, self.imageSize, R, t, alpha = rectify_scale)
                left_maps = cv2.initUndistortRectifyMap(self.K,self.dist_coeff, R1, P1, self.imageSize, cv2.CV_16SC2)
                right_maps = cv2.initUndistortRectifyMap(self.K,self.dist_coeff, R2, P2, self.imageSize, cv2.CV_16SC2)
                
                
                left_img_remap = cv2.remap(I1, left_maps[0], left_maps[1], cv2.INTER_LANCZOS4)
                right_img_remap = cv2.remap(I2, right_maps[0], right_maps[1], cv2.INTER_LANCZOS4)     

                # I_show=np.hstack([left_img_remap,right_img_remap])
                I_show=right_img_remap

                I_show = cv2.resize(I_show,(int(self.imageSize[0]),int(self.imageSize[1]/2)))

                cv2.imshow("left chess", I_show)                     


                if cv2.waitKey(25) & 0xFF == ord('q'): 
                    pass 


def vgg_F_from_P(P1, P2):

    X1 = P1[[1, 2],:]
    X2 = P1[[2, 0],:]
    X3 = P1[[0, 1],:]
    Y1 = P2[[1, 2],:]
    Y2 = P2[[2, 0],:]
    Y3 = P2[[0, 1],:]

    F = [[det(X1, Y1), det(X2, Y1), det(X3, Y1)],
        [det(X1, Y2), det(X2, Y2), det(X3, Y2)],
        [det(X1, Y3), det(X2, Y3), det(X3, Y3)]]

    return np.array(F)

def det(A,B):
    return np.linalg.det(np.vstack([A,B]))                