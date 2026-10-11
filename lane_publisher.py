#!/usr/bin/env python3
"""
lane_publisher.py  —  PC-A (İşleyici Bilgisayar)
=================================================
Görev:
  1. Video kaynağını okur (dosya veya kamera).
  2. Şerit tespiti uygular (Hough Lines + renk filtresi).
  3. İşlenmiş kareyi iki şekilde ROS 2'ye yayınlar:
       /lane/processed_image             (sensor_msgs/Image   — ham)
       /lane/processed_image/compressed  (sensor_msgs/CompressedImage — JPEG)
  4. Şerit verilerini JSON olarak yayınlar → /lane/data
  5. Yerel pencerede yan yana gösterir:
       Sol  = şerit tespiti sonucu
       Sağ  = Canny maskesi + ham Hough çizgileri (debug)
  6. Ayrı "Parametreler" penceresinde trackbar ile parametre ayarı.

Kullanım:
  ros2 run <paket> lane_publisher
  ros2 run <paket> lane_publisher --ros-args \
      -p source:=0 -p fps_limit:=25 -p jpeg_quality:=75 -p show_local:=true

Bağımlılıklar:
  pip install opencv-python numpy
  sudo apt install ros-<distro>-cv-bridge ros-<distro>-image-transport
"""

import json
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String


# ── Varsayılan parametreler ───────────────────────────────────────────────────
DEFAULT_PARAMS = {
    "canny_low":   50,
    "canny_high":  150,
    "hough_thr":   45,
    "min_len":     50,
    "max_gap":     80,
    "white_l_min": 190,
    "yel_h_min":   15,
    "yel_h_max":   35,
    "roi_top":     0.58,   # float → trackbar'da ×100
    "roi_hw":      0.13,   # float → trackbar'da ×100
}

TB_WIN   = "Parametreler"
MAIN_WIN = "Lane Detection  |  Sol: Sonuc   Sag: Maske"


# ── Görüntü işleme ────────────────────────────────────────────────────────────

def color_filter(frame: np.ndarray, p: dict) -> np.ndarray:
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white = cv2.inRange(
        hls,
        np.array([0,   p["white_l_min"],  0],   np.uint8),
        np.array([255, 255,              255],   np.uint8),
    )
    yellow = cv2.inRange(
        hls,
        np.array([p["yel_h_min"], 38,  115], np.uint8),
        np.array([p["yel_h_max"], 204, 255], np.uint8),
    )
    return cv2.bitwise_and(frame, frame, mask=cv2.bitwise_or(white, yellow))


def apply_roi(gray: np.ndarray, h: int, w: int, p: dict) -> np.ndarray:
    mask = np.zeros_like(gray)
    cy = int(h * p["roi_top"])
    hw = p["roi_hw"]
    verts = np.array([[
        (int(w * 0.05),        h),
        (int(w * (0.5 - hw)), cy),
        (int(w * (0.5 + hw)), cy),
        (int(w * 0.95),        h),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, verts, 255)
    return cv2.bitwise_and(gray, mask)


def average_lines(frame: np.ndarray, lines, roi_top: float):
    h = frame.shape[0]
    left_p, right_p = [], []
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
        if s == 0:
            return None
        return (int((y_bot - b) / s), y_bot, int((y_top - b) / s), y_top)

    return to_coords(left_p), to_coords(right_p)


def process_frame(frame: np.ndarray, p: dict) -> tuple:
    """
    Dönüş: (sonuc_kare, sol_serit, sag_serit, debug_kare)
    """
    h, w = frame.shape[:2]

    # İşleme hattı
    filtered = color_filter(frame, p)
    gray     = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blurred  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges    = cv2.Canny(blurred, p["canny_low"], p["canny_high"])
    roi_mask = apply_roi(edges, h, w, p)

    lines = cv2.HoughLinesP(
        roi_mask,
        rho=1, theta=np.pi / 180,
        threshold=p["hough_thr"],
        minLineLength=p["min_len"],
        maxLineGap=p["max_gap"],
    )

    left_line, right_line = average_lines(frame, lines, p["roi_top"])

    # ── Sonuç karesi ─────────────────────────────────────────────────────────
    overlay = np.zeros_like(frame)
    if left_line and right_line:
        lx1, ly1, lx2, ly2 = left_line
        rx1, ry1, rx2, ry2 = right_line
        pts = np.array(
            [[lx1, ly1], [lx2, ly2], [rx2, ry2], [rx1, ry1]], dtype=np.int32
        )
        cv2.fillPoly(overlay, [pts], (0, 160, 0))
    for coords in (left_line, right_line):
        if coords:
            x1, y1, x2, y2 = coords
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 8)
    result = cv2.addWeighted(frame, 0.85, overlay, 0.40, 0)

    # ROI trapezi (sarı)
    cy = int(h * p["roi_top"])
    hw = p["roi_hw"]
    roi_pts = np.array([
        [int(w * 0.05),        h - 1],
        [int(w * (0.5 - hw)), cy],
        [int(w * (0.5 + hw)), cy],
        [int(w * 0.95),        h - 1],
    ], dtype=np.int32)
    cv2.polylines(result, [roi_pts], True, (255, 255, 0), 1)

    # Durum etiketi
    detected = bool(left_line and right_line)
    cv2.putText(result,
                "SERIT TESPIT EDILDI" if detected else "SERIT YOK",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0) if detected else (0, 0, 255), 2)

    # ── Debug / Maske karesi ──────────────────────────────────────────────────
    debug = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)

    # Ham Hough çizgileri — mavi
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(debug, (x1, y1), (x2, y2), (255, 80, 0), 1)

    # Ortalanmış şeritler — yeşil
    for coords in (left_line, right_line):
        if coords:
            x1, y1, x2, y2 = coords
            cv2.line(debug, (x1, y1), (x2, y2), (0, 255, 80), 3)

    # ROI sınırı — sarı
    cv2.polylines(debug, [roi_pts], True, (0, 255, 255), 1)

    # Açıklama
    cv2.putText(debug, "MASKE + KENAR",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

    return result, left_line, right_line, debug


# ── ROS 2 Node ────────────────────────────────────────────────────────────────

class LanePublisherNode(Node):

    def __init__(self):
        super().__init__("lane_publisher")

        self.declare_parameter("source",        "video.mp4")
        self.declare_parameter("fps_limit",     25)
        self.declare_parameter("resize_width",  640)
        self.declare_parameter("resize_height", 360)
        self.declare_parameter("jpeg_quality",  75)
        self.declare_parameter("show_local",    True)

        src_val         = self.get_parameter("source").get_parameter_value().string_value
        self.fps_limit  = self.get_parameter("fps_limit").get_parameter_value().integer_value
        self.resize_w   = self.get_parameter("resize_width").get_parameter_value().integer_value
        self.resize_h   = self.get_parameter("resize_height").get_parameter_value().integer_value
        self.jpeg_q     = self.get_parameter("jpeg_quality").get_parameter_value().integer_value
        self.show_local = self.get_parameter("show_local").get_parameter_value().bool_value

        self.source        = int(src_val) if src_val.isdigit() else src_val
        self.bridge        = CvBridge()
        self.params        = DEFAULT_PARAMS.copy()
        self._encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_q]

        # ── Publisher'lar ─────────────────────────────────────────────────────
        # Ham Image  → eski subscriber'larla uyumluluk
        self.img_pub = self.create_publisher(Image, "/lane/processed_image", 10)
        # CompressedImage → düşük bant genişliği (~10-30x daha az)
        self.img_pub_compressed = self.create_publisher(
            CompressedImage, "/lane/processed_image/compressed", 10
        )
        self.data_pub = self.create_publisher(String, "/lane/data", 10)

        # ── Video kaynağı ─────────────────────────────────────────────────────
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            self.get_logger().error(f"Video kaynağı açılamadı: {self.source}")
            raise RuntimeError("Video source error")

        # ── FPS sayacı ────────────────────────────────────────────────────────
        self._fps_frames = 0
        self._fps_t0     = time.perf_counter()
        self._fps_disp   = 0.0

        # ── Yerel pencereler + trackbar ───────────────────────────────────────
        if self.show_local:
            # Trackbar penceresi
            cv2.namedWindow(TB_WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(TB_WIN, 520, 320)
            self._setup_trackbars()

            # Ana görüntü penceresi (yan yana → 2× genişlik)
            cv2.namedWindow(MAIN_WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(MAIN_WIN, self.resize_w * 2 + 4, self.resize_h)

        period = 1.0 / self.fps_limit
        self.timer = self.create_timer(period, self.timer_callback)
        self.get_logger().info(
            f"LanePublisher başlatıldı | kaynak={self.source} | "
            f"{self.fps_limit} fps | {self.resize_w}x{self.resize_h} | "
            f"jpeg={self.jpeg_q} | yerel={self.show_local}"
        )

    # ── Trackbar kurulum / okuma ──────────────────────────────────────────────

    def _setup_trackbars(self):
        p = self.params
        cv2.createTrackbar("Canny Low",   TB_WIN, p["canny_low"],            255, lambda _: None)
        cv2.createTrackbar("Canny High",  TB_WIN, p["canny_high"],           255, lambda _: None)
        cv2.createTrackbar("Hough Thr",   TB_WIN, p["hough_thr"],            150, lambda _: None)
        cv2.createTrackbar("Min Len",     TB_WIN, p["min_len"],              200, lambda _: None)
        cv2.createTrackbar("Max Gap",     TB_WIN, p["max_gap"],              300, lambda _: None)
        cv2.createTrackbar("White L Min", TB_WIN, p["white_l_min"],          255, lambda _: None)
        cv2.createTrackbar("Yel H Min",   TB_WIN, p["yel_h_min"],            90,  lambda _: None)
        cv2.createTrackbar("Yel H Max",   TB_WIN, p["yel_h_max"],            90,  lambda _: None)
        cv2.createTrackbar("ROI Top x100",TB_WIN, int(p["roi_top"] * 100),  100,  lambda _: None)
        cv2.createTrackbar("ROI HW  x100",TB_WIN, int(p["roi_hw"]  * 100),   50,  lambda _: None)

    def _read_trackbars(self):
        g = lambda name: cv2.getTrackbarPos(name, TB_WIN)
        self.params["canny_low"]   = g("Canny Low")
        self.params["canny_high"]  = g("Canny High")
        self.params["hough_thr"]   = max(1, g("Hough Thr"))
        self.params["min_len"]     = max(1, g("Min Len"))
        self.params["max_gap"]     = g("Max Gap")
        self.params["white_l_min"] = g("White L Min")
        yel_min = g("Yel H Min")
        yel_max = g("Yel H Max")
        self.params["yel_h_min"]   = yel_min
        self.params["yel_h_max"]   = max(yel_min + 1, yel_max)
        self.params["roi_top"]     = max(0.10, g("ROI Top x100") / 100.0)
        self.params["roi_hw"]      = max(0.01, g("ROI HW  x100") / 100.0)

    def _draw_tb_info(self):
        """Trackbar penceresine mevcut parametre değerlerini yaz."""
        info = np.zeros((320, 520, 3), dtype=np.uint8)
        cv2.putText(info, "Aktif Parametreler",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)
        items = list(self.params.items())
        for i, (k, v) in enumerate(items):
            text = f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}"
            cv2.putText(info, text,
                        (10, 62 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(info, "[ q ] = cikis",
                    (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.imshow(TB_WIN, info)

    # ── Ana döngü ─────────────────────────────────────────────────────────────

    def timer_callback(self):
        t_start = time.perf_counter()

        ret, frame = self.cap.read()
        if not ret:
            if isinstance(self.source, str):           # video bitti → başa sar
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        frame = cv2.resize(frame, (self.resize_w, self.resize_h))

        # Trackbar'dan parametreleri güncelle
        if self.show_local:
            self._read_trackbars()

        result, left_line, right_line, debug = process_frame(frame, self.params)

        # FPS güncelle
        self._fps_frames += 1
        elapsed_total = time.perf_counter() - self._fps_t0
        if elapsed_total >= 1.0:
            self._fps_disp   = self._fps_frames / elapsed_total
            self._fps_frames = 0
            self._fps_t0     = time.perf_counter()

        proc_ms = (time.perf_counter() - t_start) * 1000
        cv2.putText(result,
                    f"FPS: {self._fps_disp:.1f}  |  {proc_ms:.1f} ms",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

        # ── Yerel gösterim: Sol = sonuç, Sağ = maske ─────────────────────────
        if self.show_local:
            sep      = np.zeros((self.resize_h, 4, 3), dtype=np.uint8)
            combined = np.hstack([result, sep, debug])
            cv2.imshow(MAIN_WIN, combined)
            self._draw_tb_info()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.get_logger().info("Kullanıcı 'q' ile durdurdu.")
                rclpy.shutdown()
                return

        # ── ROS 2 yayın ───────────────────────────────────────────────────────
        now = self.get_clock().now()

        # Ham Image (eski subscriber uyumu → önceki haliyle aynı topic)
        ros_img               = self.bridge.cv2_to_imgmsg(result, encoding="bgr8")
        ros_img.header.stamp  = now.to_msg()
        ros_img.header.frame_id = "lane_camera"
        self.img_pub.publish(ros_img)

        # CompressedImage (JPEG — bant genişliği tasarrufu)
        ok, buf = cv2.imencode(".jpg", result, self._encode_param)
        if ok:
            cmsg               = CompressedImage()
            cmsg.header.stamp  = now.to_msg()
            cmsg.header.frame_id = "lane_camera"
            cmsg.format        = "jpeg"
            cmsg.data          = buf.tobytes()
            self.img_pub_compressed.publish(cmsg)

        # JSON şerit verisi
        lane_data = {
            "timestamp":      now.nanoseconds,
            "left_lane":      list(left_line)  if left_line  else None,
            "right_lane":     list(right_line) if right_line else None,
            "lanes_detected": bool(left_line and right_line),
        }
        self.data_pub.publish(String(data=json.dumps(lane_data)))

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


# ── Giriş noktası ─────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    try:
        node = LanePublisherNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f"[HATA] {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
