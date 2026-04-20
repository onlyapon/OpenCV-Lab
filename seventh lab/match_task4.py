
import cv2
import numpy as np

def match_features(img1_path, img2_path, name1, name2):
    img1 = cv2.imread(img1_path, 0) # queryImage
    img2 = cv2.imread(img2_path, 0) # trainImage
    
    if img1 is None or img2 is None:
        print(f"Failed to load {img1_path} or {img2_path}")
        return

    # Initiate ORB detector
    orb = cv2.ORB_create()
    
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    
    if des1 is None or des2 is None:
        print(f"No descriptors found for {name1} or {name2}")
        return

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(des1,des2)
    
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    
    print(f"Matches between {name1} and {name2}: {len(matches)}")
    if len(matches) > 10:
        print("  -> Significant match found!")
        # Draw top 10 matches (optional, but we just want text output for now)
        good_matches = matches[:10]
        # Check matching coordinates to guess scale/position?
        pts1 = np.float32([ kp1[m.queryIdx].pt for m in good_matches ])
        pts2 = np.float32([ kp2[m.trainIdx].pt for m in good_matches ])
        
        # Estimate transformation?
        # A simple bounding box check
        print(f"  Avg Pos in {name2}: x={np.mean(pts2[:,0]):.1f}, y={np.mean(pts2[:,1]):.1f}")


match_features("foreground.jpeg", "4.png", "Foreground", "Result 4.png")
match_features("background.webp", "4.png", "Background", "Result 4.png")
