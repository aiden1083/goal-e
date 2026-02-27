# 功能：
# 1) 摄像头/视频输入（--video 或 --cam）
# 2) puck 圆检测 + 预测轨迹可视化（含反弹）
# 3) tune 窗口黑图刷新（解决 macOS trackbar 挤一起）
# 4) 播放结束后：选择重播(r) / 退出(q或ESC)
# 5) 播放过程中：空格暂停/继续；r 立即重播（仅视频）；n 单步下一帧；ESC 退出

import numpy as np
import cv2
import argparse
import time

from ui import setup_tune_window, read_params, clamp_roi
from vision import preprocess_gray, detect_best_circle, is_circle_like
from predictor import Bounds, simulate_trajectory


def show_end_screen(last_out):
    """视频结束后显示提示：r重播 / q或ESC退出"""
    if last_out is None:
        last_out = np.zeros((480, 640, 3), dtype=np.uint8)

    out = last_out.copy()
    cv2.putText(out, "Video ended", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(out, "Press 'r' to replay", (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(out, "Press 'q' or ESC to quit", (30, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("frame", out)

    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == ord('r'):
            return "replay"
        if k == ord('q') or k == 27:
            return "quit"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="", help="video file path; empty = webcam")
    parser.add_argument("--cam", type=int, default=0, help="webcam index (default 0)")
    args = parser.parse_args()

    source = args.video if args.video else args.cam
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"无法打开视频源: {source}")
        return

    ok, frame = cap.read()
    if not ok:
        print("无法读取第一帧。")
        cap.release()
        return

    H, W = frame.shape[:2]
    setup_tune_window(default_w=W, default_h=H)

    # ===== 预测轨迹参数（只可视化）=====
    HORIZON = 0.60   # 预测未来 0.6 秒
    STEP = 0.01      # 预测步长 0.01 秒（越小越密）
    ALPHA = 0.6      # 速度指数平滑（0~1，越大越跟随测量）

    last_puck = None
    last_t = None
    vx_s, vy_s = 0.0, 0.0

    # 播放控制
    paused = False
    step_once = False  # 按 n 单步
    last_out = None

    # 为了避免视频结束后第一帧丢失（我们已经读过一帧用于初始化）
    # 把这帧先处理一次：用一个“缓存帧”机制
    buffered_frame = frame
    use_buffered = True

    while True:
        # ===== 读帧逻辑（支持暂停/单步/重播菜单）=====
        if use_buffered:
            frame = buffered_frame
            use_buffered = False
            ok = True
        else:
            if not paused or step_once:
                ok, frame = cap.read()
                step_once = False
            else:
                ok = True  # 暂停时沿用上一帧画面
                frame = frame  # noqa: B018  (保持当前 frame)

        if not ok:
            # 视频结束：弹出菜单；摄像头则直接退出
            if args.video:
                choice = show_end_screen(last_out)
                if choice == "replay":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    paused = False
                    step_once = False
                    # 重播后把速度/历史清掉，避免轨迹乱飞
                    last_puck = None
                    last_t = None
                    vx_s, vy_s = 0.0, 0.0
                    continue
                else:
                    break
            else:
                break

        H, W = frame.shape[:2]
        p = read_params()

        # ROI
        if p["useROI"]:
            rx, ry, rw, rh = clamp_roi(p["roi"], W, H)
            frame_use = frame[ry:ry + rh, rx:rx + rw]
            offset_x, offset_y = rx, ry
        else:
            frame_use = frame
            offset_x, offset_y = 0, 0

        # 预处理
        gray_whole = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, gray_blur_use = preprocess_gray(frame_use, p["blur"])

        # 霍夫圆候选（坐标在 ROI 内）
        circles = detect_best_circle(
            gray_blur_use,
            dp=p["dp"],
            minDist=p["minDist"],
            param1=p["param1"],
            param2=p["param2"],
            minR=p["minR"],
            maxR=p["maxR"]
        )

        out = frame.copy()

        # 画ROI框
        if p["useROI"]:
            cv2.rectangle(out, (offset_x, offset_y), (offset_x + rw, offset_y + rh), (255, 255, 0), 2)
            cv2.putText(out, "ROI", (offset_x + 5, offset_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 选一个最像 puck 的圆（按圆度过滤）
        best = None
        for (x, y, r) in circles:
            cx, cy = x + offset_x, y + offset_y
            if not is_circle_like(gray_whole, cx, cy, r, circ_thresh=p["circ_thresh"]):
                continue
            best = (cx, cy, r)
            break

        # ===== 画检测结果 + 预测轨迹 =====
        if best is not None:
            cx, cy, r = best

            # 画 puck 圆
            cv2.circle(out, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(out, f"center=({cx},{cy}) r={r}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            now = time.time()

            # 速度估计：差分 + 指数平滑
            if last_puck is not None and last_t is not None:
                dt = max(1e-3, now - last_t)
                vx_meas = (cx - last_puck[0]) / dt
                vy_meas = (cy - last_puck[1]) / dt
                vx_s = ALPHA * vx_meas + (1 - ALPHA) * vx_s
                vy_s = ALPHA * vy_meas + (1 - ALPHA) * vy_s

            last_puck = (cx, cy)
            last_t = now

            # 预测轨迹（带反弹）
            # 预测边界：跟随 ROI（如果启用 ROI，就在 ROI 边界内反弹）
            if p["useROI"]:
                # ROI 在全图坐标系下：左上角(offset_x, offset_y)，尺寸(rw, rh)
                b = Bounds(
                    xmin=float(offset_x),
                    xmax=float(offset_x + rw - 1),
                    ymin=float(offset_y),
                    ymax=float(offset_y + rh - 1),
                )
            else:
                b = Bounds(xmin=0.0, xmax=float(W - 1), ymin=0.0, ymax=float(H - 1))

            traj = simulate_trajectory(
                puck_xy=(cx, cy),          # 注意：这里用的是全图坐标 (cx,cy)
                puck_vxy=(vx_s, vy_s),
                bounds=b,
                horizon=HORIZON,
                step=STEP
            )

            # 画轨迹：点 + 线（避免太密：每隔一个点画一次）
            prev = (int(cx), int(cy))
            for i, (tx, ty) in enumerate(traj):
                if i % 2 != 0:
                    continue
                pt = (int(tx), int(ty))
                cv2.circle(out, pt, 2, (255, 0, 0), -1)   # 蓝点
                cv2.line(out, prev, pt, (255, 0, 0), 1)  # 蓝线
                prev = pt

            # 显示速度（调试）
            cv2.putText(out, f"v=({vx_s:.0f},{vy_s:.0f}) px/s",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        else:
            cv2.putText(out, "no circle",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 没检测到 puck 的时候，速度衰减，避免轨迹乱飞
            vx_s *= 0.9
            vy_s *= 0.9

        # ===== 显示播放状态提示 =====
        if paused:
            cv2.putText(out, "PAUSED (space to resume, n step)", (10, H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ===== 显示窗口 =====
        cv2.imshow("frame", out)
        cv2.imshow("gray_blur_roi", gray_blur_use)

        # 用黑图刷新 tune 布局（解决 macOS 滑条挤一起问题）
        cv2.imshow("tune", np.zeros((80, 900, 3), dtype=np.uint8))

        last_out = out

        # ===== 按键控制 =====
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord(' '):  # 空格：暂停/继续
            paused = not paused
        elif key == ord('n'):  # n：单步下一帧（暂停时也可用）
            paused = True
            step_once = True
        elif key == ord('r') and args.video:  # r：立即重播（仅视频）
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            paused = False
            step_once = False
            last_puck = None
            last_t = None
            vx_s, vy_s = 0.0, 0.0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()