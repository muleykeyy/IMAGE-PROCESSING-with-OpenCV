import cv2
import matplotlib.pyplot as plt

chocoS=cv2.imread("chocolates.jpg",0)
plt.figure()
plt.imshow(chocoS,cmap="gray")
plt.axis("off")
plt.show()

cho=cv2.imread("nestle.jpg",0)
plt.figure()
plt.imshow(cho,cmap="gray")
plt.axis("off")
plt.show()

# FEATURE MATCHING ORB ALGORITHM

# ORB detector stands for Oriented Fast and Rotated Brief, this is free of cost algorithm, 
# the benefit of this algorithm is that it does not require GPU it can compute on normal CPU.
# ORB is basically the combination of two algorithms involved FAST and BRIEF where FAST stands for Features from Accelerated Segments Test whereas BRIEF stands for Binary Robust Independent Elementary Features.
# ORB detector first uses FAST algorithm, this FAST algorithm finds the key points then applies Harries corner measure to find top N numbers of key points among them, 
# this algorithm quickly selects the key points by comparing the distinctive regions like the intensity variations.
# This algorithm works on Key point matching, Key point is distinctive regions in an image like the intensity variations.
# Now the role of BRIEF algorithm comes, this algorithm takes the key points and turn into the binary descriptor/binary feature vector that contains the combination of 0s and1s only. 
# The key points founded by FAST algorithm and Descriptors created by BRIEF algorithm both together represent the object. 
# BRIEF is the faster method for feature descriptor calculation and it also provides a high recognition rate until and unless there is large in-plane rotation.



# Detects edges, corners ..features
orb = cv2.ORB_create()
kp1,des1=orb.detectAndCompute(cho,None)
kp2,des2=orb.detectAndCompute(chocoS,None)

# matcher
bf=cv2.BFMatcher(cv2.NORM_HAMMING)

matches=bf.match(des1,des2)

matches=sorted(matches,key=lambda x:x.distance)

plt.figure()
img_match=cv2.drawMatches(cho,kp1,chocoS,kp2,matches[:20],None,flags=2)
plt.imshow(img_match)
plt.title("ORB")
plt.axis("off")

# FEATURE MATCHING SIFT ALGORITHM (pip install opencv-contrib-python --user)

# SIFT (Scale Invariant Fourier Transform) Detector is used in the detection of interest points on an input image. 
# It allows identification of localized features in images which is essential in applications such as: 
 
            # Object Recognition in Images
            # Path detection and obstacle avoidance algorithms
            # Gesture recognition, Mosaic generation, etc
sift=cv2.SIFT_create()

# BRUTE FORCE
bf=cv2.BFMatcher()

kp1,des1=sift.detectAndCompute(cho,None)
kp2,des2=sift.detectAndCompute(chocoS,None)

matches=bf.knnMatch(des1,des2,k=2)
nice_match=[]

for match1,match2 in matches:
    
    if match1.distance < 0.75*match2.distance:
        nice_match.append([match1])

plt.figure()
sift_matches=cv2.drawMatchesKnn(cho,kp1,chocoS,kp2,nice_match,None,flags=2)

plt.imshow(sift_matches)
plt.title("SIFT")
plt.axis("off")    