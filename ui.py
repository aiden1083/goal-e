# ui.py
import cv2

STEP = 10  # ROI 滑条每一格代表 10 像素

def nothing(x):
    pass

def setup_tune_window(default_w=1280, default_h=720):
    # 建议加宽一点
    cv2.namedWindow("tune", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow("tune", 900, 950)

    # ===== 图像预处理 / Hough 圆参数 =====
    cv2.createTrackbar("blur", "tune", 7, 31, nothing)
    cv2.createTrackbar("dp_x10", "tune", 12, 30, nothing)
    cv2.createTrackbar("minDist", "tune", 60, 600, nothing)
    cv2.createTrackbar("param1", "tune", 120, 300, nothing)
    cv2.createTrackbar("param2", "tune", 45, 200, nothing)
    cv2.createTrackbar("minR", "tune", 10, 400, nothing)
    cv2.createTrackbar("maxR", "tune", 80, 600, nothing)
    cv2.createTrackbar("circ_x100", "tune", 80, 100, nothing)

    # ===== ROI（缩放滑条：值 * STEP 才是真实像素）=====
    cv2.createTrackbar("useROI", "tune", 0, 1, nothing)

    # 0..300 代表 0..3000 像素
    cv2.createTrackbar("roi_x10", "tune", 0, 300, nothing)
    cv2.createTrackbar("roi_y10", "tune", 0, 300, nothing)

    # 宽高上限按 4000/10=400（你也可以按需要改大）
    cv2.createTrackbar("roi_w10", "tune", max(1, default_w // STEP), 400, nothing)
    cv2.createTrackbar("roi_h10", "tune", max(1, default_h // STEP), 400, nothing)

def read_params():
    blur = cv2.getTrackbarPos("blur", "tune")
    dp = cv2.getTrackbarPos("dp_x10", "tune") / 10.0
    minDist = cv2.getTrackbarPos("minDist", "tune")
    param1 = cv2.getTrackbarPos("param1", "tune")
    param2 = cv2.getTrackbarPos("param2", "tune")

    minR = cv2.getTrackbarPos("minR", "tune")
    maxR = cv2.getTrackbarPos("maxR", "tune")
    if maxR < minR:
        maxR = minR

    circ_thresh = cv2.getTrackbarPos("circ_x100", "tune") / 100.0

    useROI = cv2.getTrackbarPos("useROI", "tune") == 1

    # ROI 缩放读回（乘 STEP 还原为像素）
    rx = cv2.getTrackbarPos("roi_x10", "tune") * STEP
    ry = cv2.getTrackbarPos("roi_y10", "tune") * STEP
    rw = cv2.getTrackbarPos("roi_w10", "tune") * STEP
    rh = cv2.getTrackbarPos("roi_h10", "tune") * STEP

    return {
        "blur": blur,
        "dp": dp,
        "minDist": minDist,
        "param1": param1,
        "param2": param2,
        "minR": minR,
        "maxR": maxR,
        "circ_thresh": circ_thresh,
        "useROI": useROI,
        "roi": (rx, ry, rw, rh),
    }

def clamp_roi(roi, frame_w, frame_h):
    rx, ry, rw, rh = roi
    rx = max(0, min(rx, frame_w - 1))
    ry = max(0, min(ry, frame_h - 1))
    rw = max(1, min(rw, frame_w - rx))
    rh = max(1, min(rh, frame_h - ry))
    return rx, ry, rw, rh