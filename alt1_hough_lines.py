"""
Alternatif 1 — Klasik Serit Tespiti (Hough Lines) + Canli Trackbar Ayarlari

Sol panel : Tespit sonucu (seritin cizildigi orijinal kare)
Sag panel : Renk maskesi (algoritmanin "gordugu" pikseller)

Kullanim:
  python alt1_hough_lines.py
  python alt1_hough_lines.py 1
  python alt1_hough_lines.py video.mp4
"""

import sys
import json
import time
import cv2
import numpy as np

# ── ROS 2 opsiyonel bagimliliklari ───────────────────────────────────────────
from typing import Any as _Any

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
PROC_W, PROC_H = 1024, 576         # tum girisler bu boyuta getirilir
COMBINED_W     = PROC_W * 2        # yan yana toplam genislik

CTRL_WIN = "Alt1 - Ayarlar"
VIEW_WIN = "Alt1 | TESPIT  <-->  MASKE"


# ── Trackbar Kontrol Paneli ───────────────────────────────────────────────────

def create_controls() -> None:
    cv2.namedWindow(CTRL_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CTRL_WIN, 440, 340)
    noop = lambda _: None

    cv2.createTrackbar("Canny Alt Esik",      CTRL_WIN,  50, 255, noop)
    cv2.createTrackbar("Canny Ust Esik",      CTRL_WIN, 150, 255, noop)
    cv2.createTrackbar("Hough Oy Esigi",      CTRL_WIN,  50, 200, noop)
    cv2.createTrackbar("Min Cizgi Uzunlugu",  CTRL_WIN,  40, 300, noop)
    cv2.createTrackbar("Max Bosluk",          CTRL_WIN, 100, 400, noop)
    cv2.createTrackbar("Beyaz L Min",         CTRL_WIN, 200, 255, noop)
    cv2.createTrackbar("Sari H Min",          CTRL_WIN,  15,  50, noop)
    cv2.createTrackbar("Sari H Max",          CTRL_WIN,  35,  50, noop)
    cv2.createTrackbar("ROI Ust Y %",         CTRL_WIN,  58,  88, noop)
    cv2.createTrackbar("ROI Ust Yaricap %",   CTRL_WIN,  13,  45, noop)


def read_params() -> dict:
    g     = lambda k: cv2.getTrackbarPos(k, CTRL_WIN)
    h_min = g("Sari H Min")
    h_max = g("Sari H Max")
    return {
        "canny_low":   g("Canny Alt Esik"),
        "canny_high":  g("Canny Ust Esik"),
        "hough_thr":   max(1, g("Hough Oy Esigi")),
        "min_len":     max(1, g("Min Cizgi Uzunlugu")),
        "max_gap":     max(1, g("Max Bosluk")),
        "white_l_min": g("Beyaz L Min"),
        "yel_h_min":   h_min,
        "yel_h_max":   max(h_min + 1, h_max),
        "roi_top":     g("ROI Ust Y %")     / 100.0,
        "roi_hw":      g("ROI Ust Yaricap %") / 100.0,
    }


# ── Goruntu Isleme ────────────────────────────────────────────────────────────

def color_filter(frame: np.ndarray, p: dict) -> np.ndarray:
    hls    = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white  = cv2.inRange(hls,
                         np.array([0,          p["white_l_min"], 0],   np.uint8),
                         np.array([255,        255,              255],  np.uint8))
    yellow = cv2.inRange(hls,
                         np.array([p["yel_h_min"], 38,  115], np.uint8),
                         np.array([p["yel_h_max"], 204, 255], np.uint8))
    mask = cv2.bitwise_or(white, yellow)
    return cv2.bitwise_and(frame, frame, mask=mask), mask


def apply_roi_mask(h: int, w: int, p: dict) -> np.ndarray:
    mask     = np.zeros((h, w), dtype=np.uint8)
    cy       = int(h * p["roi_top"])
    hw       = p["roi_hw"]
    vertices = np.array([[
        (int(w * 0.05),        h),
        (int(w * (0.5 - hw)), cy),
        (int(w * (0.5 + hw)), cy),
        (int(w * 0.95),        h),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    return mask


def average_lines(frame: np.ndarray, lines, roi_top: float):
    h                    = frame.shape[0]
    left_p, right_p      = [], []
    if lines is None:
        return None, None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.5:
            continue
        intercept = y1 - slope * x1
        (left_p if slope < 0 else right_p).append((slope, intercept))

    def to_coords(params):
        if not params:
            return None
        s, b  = np.mean(params, axis=0)
        y_bot = h
        y_top = int(h * roi_top)
        return (int((y_bot - b) / s), y_bot, int((y_top - b) / s), y_top)

    return to_coords(left_p), to_coords(right_p)


def process_frame(frame: np.ndarray, p: dict) -> tuple[np.ndarray, np.ndarray, tuple | None, tuple | None]:
    """
    Donusu:
      left_panel  — serit tespiti sonucu
      right_panel — renk maskesi (algoritmanin gordukleri)
    """
    h, w = frame.shape[:2]

    filtered, color_mask = color_filter(frame, p)
    roi_mask             = apply_roi_mask(h, w, p)
    color_roi            = cv2.bitwise_and(color_mask, roi_mask)

    gray    = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, p["canny_low"], p["canny_high"])
    roi_edges = cv2.bitwise_and(edges, roi_mask)

    lines                = cv2.HoughLinesP(roi_edges,
                                           rho=1, theta=np.pi / 180,
                                           threshold=p["hough_thr"],
                                           minLineLength=p["min_len"],
                                           maxLineGap=p["max_gap"])
    left_line, right_line = average_lines(frame, lines, p["roi_top"])

    # ── Sol Panel: Tespit Sonucu ──────────────────────────────────────────────
    overlay = np.zeros_like(frame)
    if left_line and right_line:
        lx1, ly1, lx2, ly2 = left_line
        rx1, ry1, rx2, ry2 = right_line
        pts = np.array([[lx1, ly1], [lx2, ly2],
                        [rx2, ry2], [rx1, ry1]], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 160, 0))
    for coords in [left_line, right_line]:
        if coords:
            x1, y1, x2, y2 = coords
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 6)

    left_panel = cv2.addWeighted(frame, 0.85, overlay, 0.40, 0)

    # ROI trapezi
    cy = int(h * p["roi_top"])
    hw = p["roi_hw"]
    roi_pts = np.array([
        [int(w * 0.05),        h - 1],
        [int(w * (0.5 - hw)), cy],
        [int(w * (0.5 + hw)), cy],
        [int(w * 0.95),        h - 1],
    ], dtype=np.int32)
    cv2.polylines(left_panel, [roi_pts], True, (255, 255, 0), 1)

    # Bilgi kutusu
    lines_found = left_line is not None or right_line is not None
    status_col  = (0, 255, 0) if lines_found else (0, 0, 255)
    info = [
        f"Canny: {p['canny_low']}-{p['canny_high']}",
        f"Hough: esik={p['hough_thr']} uzun={p['min_len']} bos={p['max_gap']}",
        f"Beyaz L>={p['white_l_min']}  Sari H={p['yel_h_min']}-{p['yel_h_max']}",
        f"ROI: Y={int(p['roi_top']*100)}%  hw={int(p['roi_hw']*100)}%",
    ]
    for i, txt in enumerate(info):
        cv2.putText(left_panel, txt, (8, 26 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)
    cv2.putText(left_panel,
                "SERIT BULUNDU" if lines_found else "SERIT YOK",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_col, 2)

    # ── Sag Panel: Renk Maskesi ───────────────────────────────────────────────
    # Renk filtresi + ROI kesisimi → BGR renk kodlamasiyla goster
    right_panel = cv2.cvtColor(color_roi, cv2.COLOR_GRAY2BGR)
    # Tespit edilen kenar piksellerini sari ile vurgula
    edge_vis    = cv2.cvtColor(roi_edges, cv2.COLOR_GRAY2BGR)
    edge_vis[:, :, 0] = 0                          # mavi kanal sifirla
    right_panel = cv2.addWeighted(right_panel, 0.6, edge_vis, 1.0, 0)

    cv2.putText(right_panel, f"Renk Maskesi + Kenar",
                (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # left_line / right_line publish sistemi icin de donduruluyor
    return left_panel, right_panel, left_line, right_line


# ── Yan Yana Birlestirme ─────────────────────────────────────────────────────

def make_combined(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Sol ve sag panelleri etiketleyip yatay birlestir."""
    for img, label in [(left, " TESPIT "), (right, " MASKE ")]:
        tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
        x = img.shape[1] - tw - 8
        cv2.rectangle(img, (x - 4, 2), (x + tw + 4, th + 8), (0, 0, 0), -1)
        cv2.putText(img, label, (x, th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

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

    # Video FPS'ini oku; webcam veya bilinmiyorsa 30 varsayilan
    src_fps  = cap.get(cv2.CAP_PROP_FPS)
    wait_ms  = max(1, int(1000 / src_fps)) if src_fps > 1 else 1

    create_controls()
    cv2.namedWindow(VIEW_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VIEW_WIN, 1600, 450)   # buyuk kombinasyonu ekrana sigidir

    print(f"[BILGI] Cozunurluk: {PROC_W}x{PROC_H} | FPS: {src_fps:.1f} | "
          f"kare gecikmesi: {wait_ms} ms")
    print("[BILGI] Cikmak icin 'q' tusuna basin.")

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        frame = cv2.resize(frame, (PROC_W, PROC_H))
        p     = read_params()

        left_panel, right_panel, *_ = process_frame(frame, p)
        cv2.imshow(VIEW_WIN, make_combined(left_panel, right_panel))

        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── ROS 2 Publish Sistemi ────────────────────────────────────────────────────

DEFAULT_PARAMS: dict = {
    "canny_low":   50,  "canny_high":  150,
    "hough_thr":   50,  "min_len":     40,  "max_gap": 100,
    "white_l_min": 200, "yel_h_min":   15,  "yel_h_max": 35,
    "roi_top":     0.58,"roi_hw":      0.13,
}


class LanePublisherNode(_RosNode):
    """
    Alt-1 Hough Lines → ROS 2 publisher.

    Topics:
      /lane/processed_image             sensor_msgs/Image
      /lane/processed_image/compressed  sensor_msgs/CompressedImage
      /lane/data                        std_msgs/String  (JSON)

    Calistirma:
      python alt1_hough_lines.py --ros [--source video.mp4] [--fps 25] [--no-gui]
    """

    def __init__(self, source, fps_limit: int = 25,
                 jpeg_quality: int = 75, show_local: bool = True) -> None:
        super().__init__("lane_publisher_alt1")
        self.source      = source
        self.fps_limit   = fps_limit
        self.show_local  = show_local
        self.bridge      = CvBridge()
        self._enc_param  = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        self._fps_frames = 0
        self._fps_t0     = time.perf_counter()
        self._fps_disp   = 0.0

        self.img_pub   = self.create_publisher(Image,            "/lane/processed_image",            10)
        self.img_pub_c = self.create_publisher(CompressedImage,  "/lane/processed_image/compressed", 10)
        self.data_pub  = self.create_publisher(RosString,        "/lane/data",                       10)

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Video kaynagi acilamadi: {source}")

        if show_local:
            create_controls()
            cv2.namedWindow(VIEW_WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(VIEW_WIN, 1600, 450)

        self.timer = self.create_timer(1.0 / fps_limit, self._cb)
        self.get_logger().info(
            f"[Alt1-Hough] src={source} | {fps_limit}fps | yerel={show_local}"
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

        left_panel, right_panel, left_line, right_line = process_frame(frame, p)

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
            cv2.imshow(VIEW_WIN, make_combined(left_panel, right_panel))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.get_logger().info("Kullanici 'q' ile durdurdu.")
                rclpy.shutdown()
                return

        now = self.get_clock().now()

        # sensor_msgs/Image
        msg                 = self.bridge.cv2_to_imgmsg(left_panel, encoding="bgr8")
        msg.header.stamp    = now.to_msg()
        msg.header.frame_id = "lane_camera"
        self.img_pub.publish(msg)

        # sensor_msgs/CompressedImage
        ok, buf = cv2.imencode(".jpg", left_panel, self._enc_param)
        if ok:
            cmsg                  = CompressedImage()
            cmsg.header.stamp     = now.to_msg()
            cmsg.header.frame_id  = "lane_camera"
            cmsg.format           = "jpeg"
            cmsg.data             = buf.tobytes()
            self.img_pub_c.publish(cmsg)

        # std_msgs/String  (JSON serit verisi)
        self.data_pub.publish(RosString(data=json.dumps({
            "timestamp_ns":   now.nanoseconds,
            "algorithm":      "hough_lines",
            "left_lane":      list(left_line)  if left_line  else None,
            "right_lane":     list(right_line) if right_line else None,
            "lanes_detected": bool(left_line and right_line),
            "fps":            round(self._fps_disp, 1),
        })))

    def destroy_node(self) -> None:
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main_ros() -> None:
    """ROS 2 modunda baslatir.  python alt1_hough_lines.py --ros [...]"""
    if not _HAS_ROS:
        print("[HATA] rclpy / cv_bridge yuklu degil.")
        print("  Standart mod: python alt1_hough_lines.py [kaynak]")
        return

    import argparse
    ap = argparse.ArgumentParser(prog="alt1_hough_lines (ROS modu)")
    ap.add_argument("--source",  default="0",  help="Kamera no veya video yolu")
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
