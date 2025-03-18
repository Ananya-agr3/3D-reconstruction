"""
Homework 5
Submission Functions
"""
import numpy as np
import cv2
from scipy.signal import convolve2d

"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2, M):
    
    def normalize_points(pts,M):
        # Transformation matrix for translation
        T=np.array([
            [1/M,0,0],     # normalizing x
            [0,1/M,0],     # normalizing y  
            [0,0,1]                       # homogeneous coordinate remains 1
        ])

        # Adding a column of 1 to pts  for homogeneous coordinate 
        pts_final=np.column_stack((pts,np.ones(len(pts))))

        # Applying final transformation 
        pts_norm = (T @ pts_final.T).T  

        return pts_norm[:,:2] ,T
    
    def compute_fundamental_matrix(pts1,pts2):
        """Compute the Fundamental Matrix using the Eight-Point Algorithm."""
        # 1. Normalize points
        pts1_norm,T1=normalize_points(pts1,M)
        pts2_norm,T2=normalize_points(pts2,M)

        # Construct matrix A (dim= n x 9)
        A=np.array([
            [x1*x2,x1*y2,x1,y1*x2,y1*y2,y1,x2,y2,1]
            for(x1,y1),(x2,y2) in zip(pts1_norm,pts2_norm) 
        ])

        # Applying SVD on A to solve Af=0
        _,_,V=np.linalg.svd(A)
        F = V[-1].reshape(3, 3)  # Last row of V gives the solution

        # Enforce rank-2 constraint on F
        U, S, Vt = np.linalg.svd(F)
        S[-1] = 0  # Set the smallest singular value to zero
        F_rank2 = U @ np.diag(S) @ Vt

        # Denormalize F
        F_final = T2.T @ F_rank2 @ T1

        return F_final  # Normalize F so that F[2,2] = 1

    F=compute_fundamental_matrix(pts1,pts2)

    return F

"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1):
    
    window_size=25

    #converting images to grayscale for comparison
    im1_gray=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY) if len(im1.shape)==3 else im1
    im2_gray=cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY) if len(im2.shape)==3 else im2

    half_window=window_size//2
    pts2=np.zeros_like(pts1) # to store corresponding points

    for i,(x1,y1) in enumerate(pts1):

        #convert to homogeneous coordinate
        coor=np.array([x1,y1,1])

        #computing epipolar lines in second image 
        l=F@coor #l=(a,b,c) where ax+by+c=0

        #generate candidate coordinates along epipolar lines
        x_candidates=np.arange(half_window, im2.shape[1]-half_window)
        y_candidates=(-l[0]*x_candidates-l[2])/l[1] # y=(-ax-c)/b

        #Removing some outside bounds points
        valid_indices=(y_candidates>half_window) & (y_candidates<im2.shape[0]-half_window)
        x_candidates=x_candidates[valid_indices].astype(int) # converting float to int
        y_candidates=y_candidates[valid_indices].astype(int) #converting float to int

        # extracting patch around(x1,y1) in image 1
        im1_patch=im1_gray[y1-half_window:y1+half_window+1,x1-half_window:x1+half_window+1]

        best_match=None
        min_ssd=float('inf')

        #comparing im1-patch with im2_patch
        for (x2,y2) in zip(x_candidates,y_candidates):
            im2_patch=im2_gray[y2-half_window:y2+half_window+1,x2-half_window:x2+half_window+1]

            if im2_patch.shape==im1_patch.shape:
                ssd=np.sum((im1_patch-im2_patch)**2)

                if ssd<min_ssd:
                    min_ssd=ssd
                    best_match=(x2,y2)
        
        if best_match:
            pts2[i]=best_match
        else:
            pts2[i]=(x1,y1)
    
    return pts2
"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):

    # E=(K2)t . F .K1
    E=K2.T @ F @ K1

    #enforcing rank 2 constraints
    U,S,Vt=np.linalg.svd(E)
    S=np.array([1,1,0])
    E=U @ np.diag(S) @ Vt

    return E

"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):

    num_pts=pts1.shape[0] ## N
    
    pts3d=np.zeros((num_pts,3)) ## dim of pts 3d is (N x 3)

    for i in range(num_pts):

        u1,v1=pts1[i] ## u1, v1 are image points of first image
        u2,v2=pts2[i] ## u2,v2 are image points of second image 

        A=np.array([
            u1*P1[2,:3]-P1[0,:3],
            v1*P1[2,:3]-P1[1,:3],
            u2*P2[2,:3]-P2[0,:3],
            v2*P2[2,:3]-P2[1,:3]
        ])

        b=np.array([
            P1[0][3]-u1*P1[2][3],
            P1[1][3]-v1*P1[2][3],
            P2[0][3]-u2*P2[2][3],
            P2[1][3]-v2*P2[2][3]
        ])

        ## we want soln of Ax=b 
        ## (A(t)A)x=A(t)b when we take inverse of (At.A)
        ## x= (At.A)^-1 A(t)b

        pts3d[i]=np.linalg.inv(A.T @ A) @ A.T @ b
    
    return pts3d

"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    
    #computing optical centers
    c1=-np.linalg.inv(K1 @ R1)@(K1@t1)
    c2=-np.linalg.inv(K2 @ R2)@(K2@t2)

    # new x axis (r1)
    r1=(c1-c2)/np.linalg.norm(c1-c2)

    # new y using r1 and old z of left camera
    old_z=R1[2,:].T
    r2=np.cross(old_z,r1)
    r2/=np.linalg.norm(r2)

    #new z axis
    r3=np.cross(r1,r2)

    # new rotation matrix
    R=np.vstack((r1,r2,r3)).T

    # setting new R
    R1p=R2p=R

    # set new K1,k2
    K1p=K2p=K2

    # new translation vector t=-Rc
    t1p=-R1 @ c1
    t2p=-R2 @ c2

    # compute rectification matrices
    M1=(K1p @ R1p)@np.linalg.inv(K1 @ R1)
    M2=(K2p @ R2p)@np.linalg.inv(K2 @ R2)

    return M1, M2, K1p, K2p, R1p, R2p, t1p, t2p

"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp=64, win_size=25):
    
    h, w = im1.shape
    dispM = np.zeros((h, w), dtype=np.float32)
    
    half_win = (win_size - 1) // 2 
    ones_mask = np.ones((win_size, win_size))  # Mask for convolution

    for d in range(max_disp):  # Loop over disparity values
        shifted_im2 = np.roll(im2, -d, axis=1)  # Shift right image to simulate disparity
        
        # Compute sum of squared differences (SSD)
        diff = (im1 - shifted_im2) ** 2
        ssd = convolve2d(diff, ones_mask, mode='same', boundary='symm')

        # Update disparity map where SSD is smaller
        if d == 0:
            dispM = ssd
            best_disparity = np.zeros_like(ssd, dtype=np.uint8)
        else:
            update_mask = ssd < dispM
            best_disparity[update_mask] = d
            dispM[update_mask] = ssd[update_mask]

    return best_disparity

"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    
    # baseline
    b=np.linalg.norm((t1-t2).flatten())
    # focal length
    f= K1[0,0]

    # Avoid division by zero: set depth to 0 where disparity is 0
    depthM = np.zeros_like(dispM, dtype=np.float32)
    valid_disp = dispM > 0  # Only compute depth where disparity is non-zero

    depthM[valid_disp]=(b*f)/dispM[valid_disp]

    return depthM

"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # replace pass by your implementation
    pass


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # replace pass by your implementation
    pass
