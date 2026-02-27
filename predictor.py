# predictor.py
from dataclasses import dataclass

@dataclass
class Bounds:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

def predict_step_with_bounce(x, y, vx, vy, dt, b: Bounds):
    """匀速 + 撞边界反弹（镜面反射）"""
    x2 = x + vx * dt
    y2 = y + vy * dt

    # x 反弹
    if x2 < b.xmin:
        x2 = b.xmin + (b.xmin - x2)
        vx = -vx
    elif x2 > b.xmax:
        x2 = b.xmax - (x2 - b.xmax)
        vx = -vx

    # y 反弹
    if y2 < b.ymin:
        y2 = b.ymin + (b.ymin - y2)
        vy = -vy
    elif y2 > b.ymax:
        y2 = b.ymax - (y2 - b.ymax)
        vy = -vy

    return x2, y2, vx, vy

def simulate_trajectory(puck_xy, puck_vxy, bounds: Bounds, horizon=0.6, step=0.01):
    """
    返回未来轨迹点列表：[(x,y), ...]
    horizon: 预测时长（秒）
    step: 每步时间（秒）
    """
    x, y = puck_xy
    vx, vy = puck_vxy
    pts = []

    t = 0.0
    while t <= horizon:
        x, y, vx, vy = predict_step_with_bounce(x, y, vx, vy, step, bounds)
        pts.append((x, y))
        t += step
    return pts