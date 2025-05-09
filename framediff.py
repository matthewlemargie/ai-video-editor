import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# preferred
def is_shot_change(frame1, frame2, threshold=30):
    diff = cv2.absdiff(frame1, frame2)
    mean_diff = np.mean(diff)
    return mean_diff > threshold

# not preferred
def is_shot_change_hist(frame1, frame2, threshold=0.6):
    hist1 = cv2.calcHist([cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return correlation < threshold

# slow but works
def is_shot_change_ssim(frame1, frame2, threshold=0.8):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score < threshold
