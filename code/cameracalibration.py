import numpy as np
import cv2
import glob

def get_undistort_params(fn_prefix, nx=9, ny=6):
  images = glob.glob(fn_prefix)
  #print(images)
  objpoints = []
  imgpoints = []
  objp = np.zeros((nx*ny, 3), np.float32)
  objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

  for fn in images:
    img = cv2.imread(fn)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, corners = cv2.findChessboardCorners(img_gray, (nx, ny), None)

    if retval:
      objpoints.append(objp)
      imgpoints.append(corners)

  ret, camMat, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                              imgpoints,
                                                              img.shape[:-1][::-1],
                                                              None, None)
  return ret, camMat, distCoeffs, rvecs, tvecs