import cv2
import numpy as np

pred = cv2.imread('./FCS_b0.png', -1)[:,:,0]
label = cv2.imread('../data/label/test.png', -1)
height, width = pred.shape

pred[pred > 0] = 1
label[label > 0] = 1

label_plus_pred = label + pred
label_pred = label - pred

TP = np.zeros(pred.shape)
FP = np.zeros(pred.shape)
FN = np.zeros(pred.shape)
TN = np.zeros(pred.shape)

TP[label_plus_pred == 2] = 1
FP[label_pred == -1] = 1
FN[label_pred == 1] = 1
TN[label_plus_pred == 0] = 1

tp = TP.sum()
fp = FP.sum()
fn = FN.sum()
tn = TN.sum()

m_pixacc = float(np.sum(pred == label)) / float(
    height * width
)

smooth = 0.00001
iou = tp / (tp + fp + fn + smooth)

precision = tp / (tp + fp + smooth)
recall = tp / (tp + fn + smooth)

f1_score = 2.0 * precision * recall / (precision + recall + smooth)
print("pix acc: ", m_pixacc)
print("tp: ", tp)
print("fp: ", fp)
print("tn: ", tn)
print("fn: ", fn)
print("iou: ", iou)
print("f1_score: ", f1_score)