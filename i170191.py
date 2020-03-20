import cv2
import numpy as np
    
source = cv2.imread('source.jpg')
display_image = source.copy()
target_image = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
height, width = target_image.shape
total_area = height * width
template_image = cv2.imread('template.jpg')
template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

while (1):
    orb_target = cv2.ORB_create(nfeatures=400000)
    kp2, des2 = orb_target.detectAndCompute(target_image, None)
    
    
    
    orb_template = cv2.ORB_create(nfeatures=25000)
    kp1, des1 = orb_template.detectAndCompute(template_image,None)
    
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
    search_params = dict(checks = 32)
                
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    good = []
      
    matches = flann.knnMatch(des1,des2,k=2)
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append(m)
    
    if len(good)>10:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                   
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0, maxIters = 100, confidence = 0.6)
        
        
        load_img = cv2.imread('template.jpg')            
        h,w,c = load_img.shape
        pts = np.float32([ [1,1],[1,h-1],[w-1,h-1],[w-1,1] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)
    
        dst = np.squeeze(dst)
    
        y1 = dst[0,1] 
        y2 = dst[2,1]
        x1 = dst[0,0] 
        x2 = dst[2,0]
    
        cv2.rectangle(display_image, (x1,y1), (x2,y2), color=(255,0,0), thickness=5)
        cv2.rectangle(target_image, (x1,y1), (x2,y2), color=(255,0,0), thickness=-1) 
    else:
        break
    
    cv2.imwrite('./display_box.jpg', display_image)