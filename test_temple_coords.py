import numpy as np
import helper as help
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt
import cv2

# 1. Load the two temple images and the points from data/some_corresp.npz
corresp_data = np.load("../data/some_corresp.npz")

pts1=corresp_data['pts1']
pts2=corresp_data['pts2']

im1=cv2.imread("../data/im1.png")
im2=cv2.imread("../data/im2.png")
M=max(im1.shape[0],im1.shape[1])

# 2. Run eight_point to compute F
F=sub.eight_point(pts1,pts2,M)
#help.displayEpipolarF(im1,im2,F)

# 3. Load points in image 1 from data/temple_coords.npz and Run epipolar_correspondences to get points in image 2

temple_corre=np.load("../data/temple_coords.npz")
pts1_new=temple_corre['pts1']
pts2_new=sub.epipolar_correspondences(im1,im2,F,pts1_new)
#help.epipolarMatchGUI(im1,im2,F)

# 4. Load data/intrinsics.npz and compute the essential matrix E.
camera=np.load("../data/intrinsics.npz")

K1=camera['K1']
K2=camera['K2']
# essential matrix
E=sub.essential_matrix(F,K1,K2)

# 5. Compute the camera projection matrix P1
# P1=K1.[I|0]
I=np.eye(3)  # 3 x 3
zero_vec=np.zeros((3,1))  # 3x1

# extrinsic matrix = [I|0]
M1=np.hstack((I,zero_vec))

P1=K1 @ M1

# 6. Use camera2 to get 4 camera projection matrices P2
M2s=help.camera2(E)

P2s=[]
for i in range(M2s.shape[2]):
    P2s.append(K2 @ M2s[:,:,i])

P2s=np.array(P2s)
##print(P2s)

# 7. Run triangulate using the projection matrices
candidate=[]
for i in range(len(P2s)):
    candidate.append(sub.triangulate(P1,pts1_new,P2s[i],pts2_new))

candidate=np.array(candidate)

# 8. Figure out the correct P2
## we want the P2 for which maximum number of 3D points have positive depth 
P2=None
pts3d=None
best=float('-inf')
pos=0
for j in range(candidate.shape[0]):
    pos=0
    for i in range(candidate[j].shape[0]):
        if candidate[j,i,2]>0:
            pos+=1
    
    if best<pos:
        best=pos
        pts3d=candidate[j]
        P2=P2s[j]
# print(P2)
# print(pts3d.shape)

## calculating reprojection error 

ones=np.ones((pts3d.shape[0],1))
pts4d=np.hstack((pts3d,ones))

X_proj=P1 @ pts4d.T
x_proj=(X_proj[:2]/X_proj[2]).T # converting to inhomogeneous coordinates

errors= np.linalg.norm(x_proj-pts1_new, axis=1)

reprojection_error=np.mean(errors)
print(f"The reprojection error is :{reprojection_error}")


# 9. Scatter plot the correct 3D points
fig=plt.figure(figsize=(10,7))
ax=fig.add_subplot(111,projection='3d')

ax.scatter(pts3d[:,0],pts3d[:,1],pts3d[:,2],c=pts3d[:,2],cmap='jet',marker='o')

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title("Triangulated 3D Points")

#ax.set_box_aspect([1, 1, 1]) 
ax.set_xlim(-1,2)
ax.set_ylim(-1,2)
ax.set_zlim(2,5)
plt.show()

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz

## we have K1,K2,P1,P2
M1=P1[:,:3] # starting 3 x 3
T1=P1[:,3]  # lats column
M2=P2[:,:3] # starting 3 x 3
T2=P2[:,3]  # lats column

R1=np.linalg.inv(K1) @ M1
R2=np.linalg.inv(K2) @ M2

# Ensure R is a valid rotation matrix using SVD
U, _, Vt = np.linalg.svd(R1)
R1 = U @ Vt  # This ensures R1 is orthogonal

U, _, Vt = np.linalg.svd(R2)
R2 = U @ Vt  # This ensures R2 is orthogonal

t1=np.linalg.inv(K1) @ T1
t2=np.linalg.inv(K2) @ T2

np.savez("../data/extrinsics.npz", R1=R1, R2=R2, t1=t1, t2=t2)
print("Extrinsic parameters saved to data/extrinsics.npz")