# vision.py
import cv2
import numpy as np

def preprocess_gray(frame_bgr, blur_k: int):
    """BGR->Gray + GaussianBlur"""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    blur_k = max(1, int(blur_k))
    if blur_k % 2 == 0:
        blur_k += 1

    gray_blur = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    return gray, gray_blur

def is_circle_like(gray, cx, cy, r, circ_thresh=0.80):
    """
    用轮廓圆度过滤假圆：
    circularity = 4*pi*area / perimeter^2
    """
    pad = int(r * 1.3)
    h, w = gray.shape[:2]
    x1, y1 = max(0, cx - pad), max(0, cy - pad)
    x2, y2 = min(w, cx + pad), min(h, cy + pad)

    patch = gray[y1:y2, x1:x2]
    if patch.size == 0:
        return False

    patch_blur = cv2.GaussianBlur(patch, (5, 5), 0)
    _, bw = cv2.threshold(patch_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 让目标更可能是白色：白色占比太少则反转
    if (bw > 0).mean() < 0.2:
        bw = cv2.bitwise_not(bw)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 50:
        return False

    peri = cv2.arcLength(c, True)
    if peri <= 1e-6:
        return False

    circularity = 4 * np.pi * area / (peri * peri)
    return circularity >= circ_thresh

def detect_best_circle(gray_blur, dp, minDist, param1, param2, minR, maxR):
    """HoughCircles + 返回所有候选(未加ROI偏移)"""
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=max(1.0, float(dp)),
        minDist=max(1, int(minDist)),
        param1=max(1, int(param1)),
        param2=max(1, int(param2)),
        minRadius=max(0, int(minR)),
        maxRadius=max(0, int(maxR))
    )
    if circles is None:
        return []
    circles = np.round(circles[0, :]).astype(int)
    # 按半径大到小排（你也可以改成按距离中心/面积等）
    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    return circles