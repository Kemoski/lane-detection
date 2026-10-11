"""
Alternatif 2 — Gelismis Serit Tespiti (Bird's Eye View + Sliding Window)
+ Canli Trackbar Ayarlari + ROS 2 Publish Sistemi (opsiyonel)

Sol panel : Tespit sonucu (polinom overlay orijinal karede)
Sag panel : Kayan pencere gorsellestirmesi (kus bakisi ikili gorunum)

Kullanim:
  python alt2_sliding_window.py
  python alt2_sliding_window.py 1
  python alt2_sliding_window.py video.mp4

Kalibrasyon: Sag panelde serit pikselleri dikey ve paralel cizgilerde
toplanana kadar Persp trackbarlarini ayarlayin.
"""

import sys
import json
import time
import cv2
import numpy as np
from typing import Any as _Any

# ── ROS 2 opsiyonel bagimliliklari ───────────────────────────────────────────
try:
    import rclpy                                          # type: ignore[import-untyped]
    from cv_bridge import CvBridge                        # type: ignore[import-untyped]
    from rclpy.node import Node as _RosNode               # type: ignore[import-untyped]
    from sensor_msgs.msg import CompressedImage, Image    # type: ignore[import-untyped]
    from std_msgs.msg import String as RosString          # type: ignore[import-untyped]
    _HAS_ROS = True
except ImportError:
    _HAS_ROS:        bool  = False
    rclpy:           _Any  = None
    CvBridge:        _Any  = None
    _RosNode               = object
    CompressedImage: _Any  = None
    Image:           _Any  = None
    RosString:       _Any  = None

# ── Sabit Cozunurluk ─────────────────────────────────────────────────────────
PROC_W, PROC_H = 1024, 576
COMBINED_W     = PROC_W * 2

CTRL_WIN = "Alt2 - Ayarlar"
VIEW_WIN = "Alt2 | TESPIT  <-->  KUS BAKISI + PENCERELER"


# ── Trackbar Kontrol Paneli ───────────────────────────────────────────────────

def create_controls() -> None:
    cv2.namedWindow(CTRL_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CTRL_WIN, 460, 380)
    noop = lambda _: None

    cv2.createTrackbar("Beyaz L Min",        CTRL_WIN, 200, 255, noop)
    cv2.createTrackbar("Sari H Min",         CTRL_WIN,  15,  50, noop)
    cv2.createTrackbar("Sari H Max",         CTRL_WIN,  35,  50, noop)
    cv2.createTrackbar("Persp Ust Y %",      CTRL_WIN,  63,  90, noop)
    cv2.createTrackbar("Persp Ust Yaricap%", CTRL_WIN,   6,  25, noop)
    cv2.createTrackbar("Persp Alt Kenar %",  CTRL_WIN,  12,  40, noop)
    cv2.createTrackbar("Persp Alt Y %",      CTRL_WIN,  95, 100, noop)
    cv2.createTrackbar("Pencere Sayisi",     CTRL_WIN,   9,  20, noop)
    cv2.createTrackbar("Pencere Margin",     CTRL_WIN,  80, 200, noop)
    cv2.createTrackbar("Min Piksel",         CTRL_WIN,  50, 200, noop)


def read_params() -> dict:
    g     = lambda k: cv2.getTrackbarPos(k, CTRL_WIN)
    h_min = g("Sari H Min")
    h_max = g("Sari H Max")
    ust_y = max(1, g("Persp Ust Y %"))    / 100.0
    return {
        "white_l_min": g("Beyaz L Min"),
        "yel_h_min":   h_min,
        "yel_h_max":   max(h_min + 1, h_max),
        "ust_y":       ust_y,
        "ust_hw":      max(1, g("Persp Ust Yaricap%")) / 100.0,
        "alt_off":     max(1, g("Persp Alt Kenar %"))  / 100.0,
        "alt_y":       max(ust_y + 0.01, g("Persp Alt Y %") / 100.0),
        "n_windows":   max(2, g("Pencere Sayisi")),
        "margin":      max(10, g("Pencere Margin")),
        "min_pix":     max(5,  g("Min Piksel")),
    }


# ── Goruntu Isleme ────────────────────────────────────────────────────────────

def binary_threshold(frame: np.ndarray, p: dict) -> np.ndarray:
    hls    = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white  = cv2.inRange(hls,
                         np.array([0,          p["white_l_min"], 0],   np.uint8),
                         np.array([255,        255,              255],  np.uint8))
    yellow = cv2.inRange(hls,
                         np.array([p["yel_h_min"], 38,  115], np.uint8),
                         np.array([p["yel_h_max"], 204, 255], np.uint8))
    return cv2.bitwise_or(white, yellow)


def build_transform(h: int, w: int, p: dict):
    src = np.array([
        [w * (0.5 - p["ust_hw"]),  h * p["ust_y"]],
        [w * (0.5 + p["ust_hw"]),  h * p["ust_y"]],
        [w * p["alt_off"],          h * p["alt_y"]],
        [w * (1 - p["alt_off"]),   h * p["alt_y"]],
    ], dtype=np.float32)
    dst = np.array([
        [w * 0.20, 0], [w * 0.80, 0],
        [w * 0.20, h], [w * 0.80, h],
    ], dtype=np.float32)
    M    = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv, src


def sliding_window_fit(binary_warped: np.ndarray, p: dict):
    h, w    = binary_warped.shape
    hist    = np.sum(binary_warped[h // 2:, :], axis=0)
    mid     = w // 2
    lx_cur  = int(np.argmax(hist[:mid]))
    rx_cur  = int(np.argmax(hist[mid:])) + mid

    n_win   = p["n_windows"]
    margin  = p["margin"]
    min_pix = p["min_pix"]
    win_h   = h // n_win

    nz_y, nz_x           = binary_warped.nonzero()
    left_inds, right_inds = [], []

    # Kayan pencere gorseli: binary'yi BGR yap
    viz = np.dstack([binary_warped] * 3).astype(np.uint8) * 255

    for i in range(n_win):
        y_lo = h - (i + 1) * win_h
        y_hi = h - i * win_h
        xl_lo, xl_hi = lx_cur - margin, lx_cur + margin
        xr_lo, xr_hi = rx_cur - margin, rx_cur + margin

        cv2.rectangle(viz, (xl_lo, y_lo), (xl_hi, y_hi), (0, 255, 0), 2)
        cv2.rectangle(viz, (xr_lo, y_lo), (xr_hi, y_hi), (0, 255, 0), 2)

        good_l = np.where(
            (nz_y >= y_lo) & (nz_y < y_hi) &
            (nz_x >= xl_lo) & (nz_x < xl_hi))[0]
        good_r = np.where(
            (nz_y >= y_lo) & (nz_y < y_hi) &
            (nz_x >= xr_lo) & (nz_x < xr_hi))[0]

        left_inds.append(good_l)
        right_inds.append(good_r)

        if len(good_l) > min_pix:
            lx_cur = int(np.mean(nz_x[good_l]))
        if len(good_r) > min_pix:
            rx_cur = int(np.mean(nz_x[good_r]))

    left_inds  = np.concatenate(left_inds)
    right_inds = np.concatenate(right_inds)

    lx, ly = nz_x[left_inds],  nz_y[left_inds]
    rx, ry = nz_x[right_inds], nz_y[right_inds]

    viz[ly, lx] = [255, 80,  80]   # sol serit pikselleri → mavi
    viz[ry, rx] = [80,  80, 255]   # sag serit pikselleri → kirmizi

    min_pts   = 100
    left_fit  = np.polyfit(ly, lx, 2) if len(lx) >= min_pts else None
    right_fit = np.polyfit(ry, rx, 2) if len(rx) >= min_pts else None

    # Polinom egrisini kayan pencere gorseline ciz
    if left_fit is not None:
        ploty = np.linspace(0, h - 1, h)
        pts = np.column_stack([np.polyval(left_fit, ploty), ploty]).astype(np.int32)
        cv2.polylines(viz, [pts.reshape(-1, 1, 2)], False, (255, 255,   0), 3)
    if right_fit is not None:
        ploty = np.linspace(0, h - 1, h)
        pts = np.column_stack([np.polyval(right_fit, ploty), ploty]).astype(np.int32)
        cv2.polylines(viz, [pts.reshape(-1, 1, 2)], False, (0,   255, 255), 3)

    return left_fit, right_fit, viz


def draw_lane_overlay(original: np.ndarray, binary_warped: np.ndarray,
                      left_fit, right_fit, Minv: np.ndarray) -> np.ndarray:
    if left_fit is None or right_fit is None:
        return original
    h, w   = binary_warped.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    ploty  = np.linspace(0, h - 1, h)
    lx     = np.polyval(left_fit,  ploty)
    rx     = np.polyval(right_fit, ploty)
    pts_l  = np.column_stack([lx, ploty]).astype(np.int32).reshape(-1, 1, 2)
    pts_r  = np.column_stack([rx, ploty]).astype(np.int32).reshape(-1, 1, 2)[::-1]
    cv2.fillPoly(canvas, [np.concatenate([pts_l, pts_r])], (0, 160, 0))
    cv2.polylines(canvas, [pts_l],       False, (255, 0,   0), 8)
    cv2.polylines(canvas, [pts_r[::-1]], False, (0,   0, 255), 8)
    unwarped = cv2.warpPerspective(canvas, Minv, (original.shape[1], original.shape[0]))
    return cv2.addWeighted(original, 1.0, unwarped, 0.35, 0)


# ── Yan Yana Birlestirme ─────────────────────────────────────────────────────

def make_combined(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    for img, label in [(left, " TESPIT "), (right, " KUS BAKISI ")]:
        tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)[0]
        x = img.shape[1] - tw - 8
        cv2.rectangle(img, (x - 4, 2), (x + tw + 4, th + 8), (0, 0, 0), -1)
        cv2.putText(img, label, (x, th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
    divider = np.full((PROC_H, 2, 3), 80, dtype=np.uint8)
    return np.hstack([left, divider, right])


# ── Ana Dongu ─────────────────────────────────────────────────────────────────

def main() -> None:
    arg    = sys.argv[1] if len(sys.argv) > 1 else "0"
    source = int(arg) if arg.isdigit() else arg

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[HATA] Kaynak acilamadi: {source}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    wait_ms = max(1, int(1000 / src_fps)) if src_fps > 1 else 1

    create_controls()
    cv2.namedWindow(VIEW_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VIEW_WIN, 1600, 450)

    print(f"[BILGI] Cozunurluk: {PROC_W}x{PROC_H} | FPS: {src_fps:.1f} | "
          f"kare gecikmesi: {wait_ms} ms")
    print("[BILGI] Cikmak icin 'q' tusuna basin.")
    print("[BILGI] Sag panelde serit pikselleri dikey/paralel gorunmeli.")

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        frame = cv2.resize(frame, (PROC_W, PROC_H))
        p     = read_params()

        # Her kare trackbar degisebileceginden transform her seferinde hesaplanir
        M, Minv, src_pts = build_transform(PROC_H, PROC_W, p)

        binary             = binary_threshold(frame, p)
        warped             = cv2.warpPerspective(binary, M, (PROC_W, PROC_H))
        left_fit, right_fit, window_viz = sliding_window_fit(warped, p)

        # ── Sol Panel ────────────────────────────────────────────────────────
        left_panel = draw_lane_overlay(frame.copy(), warped, left_fit, right_fit, Minv)

        ordered = src_pts[[0, 1, 3, 2]].astype(np.int32)
        cv2.polylines(left_panel, [ordered.reshape(-1, 1, 2)], True, (0, 255, 255), 1)

        found      = left_fit is not None and right_fit is not None
        status_col = (0, 255, 0) if found else (0, 0, 255)
        info = [
            f"Renk: beyaz_L>={p['white_l_min']}  sari={p['yel_h_min']}-{p['yel_h_max']}",
            f"Persp: ustY={int(p['ust_y']*100)}%  hw={int(p['ust_hw']*100)}%"
            f"  altK={int(p['alt_off']*100)}%  altY={int(p['alt_y']*100)}%",
            f"Win: sayi={p['n_windows']}  margin={p['margin']}  min={p['min_pix']}",
        ]
        for i, txt in enumerate(info):
            cv2.putText(left_panel, txt, (8, 26 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
        cv2.putText(left_panel,
                    "SERIT BULUNDU" if found else "SERIT YOK",
                    (8, PROC_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_col, 2)

        # ── Sag Panel: Kayan Pencere Gorseli ─────────────────────────────────
        cv2.imshow(VIEW_WIN, make_combined(left_panel, window_viz))

        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── ROS 2 Publish Sistemi ────────────────────────────────────────────────────

DEFAULT_PARAMS: dict = {
    "white_l_min": 200, "yel_h_min": 15,  "yel_h_max": 35,
    "ust_y":  0.63, "ust_hw": 0.06, "alt_off": 0.12, "alt_y": 0.95,
    "n_windows": 9, "margin": 80,   "min_pix": 50,
}


class LanePublisherNode(_RosNode):  # type: ignore[misc]
    """
    Alt-2 Sliding Window → ROS 2 publisher.

    Topics:
      /lane/processed_image             sensor_msgs/Image
      /lane/processed_image/compressed  sensor_msgs/CompressedImage
      /lane/data                        std_msgs/String  (JSON)

    Calistirma:
      python alt2_sliding_window.py --ros [--source video.mp4] [--fps 25] [--no-gui]
    """

    def __init__(self, source, fps_limit: int = 25,
                 jpeg_quality: int = 75, show_local: bool = True) -> None:
        super().__init__("lane_publisher_alt2")
        self.source      = source
        self.show_local  = show_local
        self.bridge      = CvBridge()
        self._enc_param  = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        self._fps_frames = 0
        self._fps_t0     = time.perf_counter()
        self._fps_disp   = 0.0

        self.img_pub   = self.create_publisher(Image,           "/lane/processed_image",            10)
        self.img_pub_c = self.create_publisher(CompressedImage, "/lane/processed_image/compressed", 10)
        self.data_pub  = self.create_publisher(RosString,       "/lane/data",                       10)

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Video kaynagi acilamadi: {source}")

        if show_local:
            create_controls()
            cv2.namedWindow(VIEW_WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(VIEW_WIN, 1600, 450)

        self.timer = self.create_timer(1.0 / fps_limit, self._cb)
        self.get_logger().info(
            f"[Alt2-SlidingWindow] src={source} | {fps_limit}fps | yerel={show_local}"
        )

    def _cb(self) -> None:
        t0 = time.perf_counter()

        ret, frame = self.cap.read()
        if not ret:
            if isinstance(self.source, str):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        frame = cv2.resize(frame, (PROC_W, PROC_H))
        p     = read_params() if self.show_local else DEFAULT_PARAMS.copy()

        M, Minv, src_pts     = build_transform(PROC_H, PROC_W, p)
        binary               = binary_threshold(frame, p)
        warped               = cv2.warpPerspective(binary, M, (PROC_W, PROC_H))
        left_fit, right_fit, window_viz = sliding_window_fit(warped, p)
        left_panel           = draw_lane_overlay(frame.copy(), warped, left_fit, right_fit, Minv)

        ordered = src_pts[[0, 1, 3, 2]].astype(np.int32)
        cv2.polylines(left_panel, [ordered.reshape(-1, 1, 2)], True, (0, 255, 255), 1)

        found = left_fit is not None and right_fit is not None
        offset_px: float | None = None
        if found:
            lane_cx   = (np.polyval(left_fit, PROC_H - 1) + np.polyval(right_fit, PROC_H - 1)) / 2.0
            offset_px = float(lane_cx - PROC_W / 2.0)

        # FPS sayaci
        self._fps_frames += 1
        elapsed = time.perf_counter() - self._fps_t0
        if elapsed >= 1.0:
            self._fps_disp   = self._fps_frames / elapsed
            self._fps_frames = 0
            self._fps_t0     = time.perf_counter()

        proc_ms = (time.perf_counter() - t0) * 1000
        cv2.putText(left_panel,
                    f"FPS:{self._fps_disp:.1f}  {proc_ms:.1f}ms  [ROS]",
                    (PROC_W - 270, PROC_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1)

        if self.show_local:
            cv2.imshow(VIEW_WIN, make_combined(left_panel, window_viz))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.get_logger().info("Kullanici 'q' ile durdurdu.")
                rclpy.shutdown()
                return

        now = self.get_clock().now()

        msg                 = self.bridge.cv2_to_imgmsg(left_panel, encoding="bgr8")
        msg.header.stamp    = now.to_msg()
        msg.header.frame_id = "lane_camera"
        self.img_pub.publish(msg)

        ok, buf = cv2.imencode(".jpg", left_panel, self._enc_param)
        if ok:
            cmsg                  = CompressedImage()
            cmsg.header.stamp     = now.to_msg()
            cmsg.header.frame_id  = "lane_camera"
            cmsg.format           = "jpeg"
            cmsg.data             = buf.tobytes()
            self.img_pub_c.publish(cmsg)

        self.data_pub.publish(RosString(data=json.dumps({
            "timestamp_ns":   now.nanoseconds,
            "algorithm":      "sliding_window",
            "left_fit":       left_fit.tolist()  if left_fit  is not None else None,
            "right_fit":      right_fit.tolist() if right_fit is not None else None,
            "offset_px":      round(offset_px, 2) if offset_px is not None else None,
            "lanes_detected": found,
            "fps":            round(self._fps_disp, 1),
        })))

    def destroy_node(self) -> None:
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main_ros() -> None:
    """ROS 2 modunda baslatir.  python alt2_sliding_window.py --ros [...]"""
    if not _HAS_ROS:
        print("[HATA] rclpy / cv_bridge yuklu degil.")
        print("  Standart mod: python alt2_sliding_window.py [kaynak]")
        return

    import argparse
    ap = argparse.ArgumentParser(prog="alt2_sliding_window (ROS modu)")
    ap.add_argument("--source",  default="0")
    ap.add_argument("--fps",     type=int, default=25)
    ap.add_argument("--jpeg",    type=int, default=75)
    ap.add_argument("--no-gui",  action="store_true")
    args, _ = ap.parse_known_args()

    source = int(args.source) if args.source.isdigit() else args.source

    rclpy.init()
    try:
        node = LanePublisherNode(source, args.fps, args.jpeg, not args.no_gui)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f"[HATA] {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    if "--ros" in sys.argv:
        sys.argv.remove("--ros")
        main_ros()
    else:
        main()
