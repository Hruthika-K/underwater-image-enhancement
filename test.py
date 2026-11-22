import cv2
from skimage.color import rgb2lab
from candidate_pipelines import metrics_of
ref = cv2.imread("data/reference-890/94_img_.png")   # use the real file name
lab = rgb2lab(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
print(metrics_of(ref, ref, lab, with_uciqe=False))