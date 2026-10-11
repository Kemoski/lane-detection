"""
lane_node.py
Ana giriş noktası — lane_processor ve lane_ros_publisher'ı birleştirir.

Kullanım:
  python lane_node.py                      # Sadece OpenCV penceresiyle
  python lane_node.py --ros                # ROS 2 publish modu
  python lane_node.py --source video.mp4  # Video dosyası
  python lane_node.py --ros --source 1    # Farklı kamera + ROS
"""

import argparse
import time

import cv2
import numpy as np

from lane_processor import DEFAULT_PARAMS, process_frame

try:
    from lane_ros_publisher import LanePublisherNode
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


# ── Pencere isimleri ──────────────────────────────────────────────────────────
TB_WIN   = "Parametreler"
MAIN_WIN = "Lane Detection  |  Sol: Sonuc   Sag: Maske"

# ── Sabit ayarlar ─────────────────────────────────────────────────────────────
RESIZE_W     = 640
RESIZE_H     = 360
FPS_LIMIT    = 25
JPEG_QUALITY = 75


# ── Trackbar yönetimi ────────────────────────────────────────────────────────

def setup_trackbars(p: dict) -> None:
    cv2.namedWindow(TB_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TB_WIN, 520, 320)
    cv2.createTrackbar("Canny Low",    TB_WIN, p["canny_low"],            255, lambda _: None)
    cv2.createTrackbar("Canny High",   TB_WIN, p["canny_high"],           255, lambda _: None)
    cv2.createTrackbar("Hough Thr",    TB_WIN, p["hough_thr"],            150, lambda _: None)
    cv2.createTrackbar("Min Len",      TB_WIN, p["min_len"],              200, lambda _: None)
    cv2.createTrackbar("Max Gap",      TB_WIN, p["max_gap"],              300, lambda _: None)
    cv2.createTrackbar("White L Min",  TB_WIN, p["white_l_min"],          255, lambda _: None)
    cv2.createTrackbar("Yel H Min",    TB_WIN, p["yel_h_min"],             90, lambda _: None)
    cv2.createTrackbar("Yel H Max",    TB_WIN, p["yel_h_max"],             90, lambda _: None)
    cv2.createTrackbar("ROI Top x100", TB_WIN, int(p["roi_top"] * 100),  100, lambda _: None)
    cv2.createTrackbar("ROI HW  x100", TB_WIN, int(p["roi_hw"]  * 100),   50, lambda _: None)


def read_trackbars(p: dict) -> None:
    g = lambda name: cv2.getTrackbarPos(name, TB_WIN)
    p["canny_low"]   = g("Canny Low")
    p["canny_high"]  = g("Canny High")
    p["hough_thr"]   = max(1, g("Hough Thr"))
    p["min_len"]     = max(1, g("Min Len"))
    p["max_gap"]     = g("Max Gap")
    p["white_l_min"] = g("White L Min")
    yel_min = g("Yel H Min")
    yel_max = g("Yel H Max")
    p["yel_h_min"]   = yel_min
    p["yel_h_max"]   = max(yel_min + 1, yel_max)
    p["roi_top"]     = max(0.10, g("ROI Top x100") / 100.0)
    p["roi_hw"]      = max(0.01, g("ROI HW  x100") / 100.0)


def draw_tb_info(p: dict) -> None:
    """Trackbar penceresine mevcut parametre değerlerini yaz."""
    info = np.zeros((320, 520, 3), dtype=np.uint8)
    cv2.putText(info, "Aktif Parametreler",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)
    for i, (k, v) in enumerate(p.items()):
        text = f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}"
        cv2.putText(info, text,
                    (10, 62 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(info, "[ q ] = cikis",
                (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    cv2.imshow(TB_WIN, info)


# ── Ana döngü ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Şerit Tespit Node'u")
    parser.add_argument("--ros",    action="store_true",
                        help="ROS 2 publish modunda çalıştır")
    parser.add_argument("--source", default="0",
                        help="Kamera index veya video dosya yolu (varsayılan: 0)")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    # ── 1. ROS node (opsiyonel) ─────────────────────────────────────────────
    ros_node = None
    if args.ros:
        if not ROS_AVAILABLE:
            print("UYARI: ROS 2 bulunamadı. --ros devre dışı bırakıldı.")
        else:
            ros_node = LanePublisherNode(jpeg_quality=JPEG_QUALITY)
            print("ROS 2 publish modu aktif.")
            print("  /lane/processed_image")
            print("  /lane/processed_image/compressed")
            print("  /lane/data")

    # ── 2. Video kaynağı ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"HATA: Video kaynağı açılamadı: {source}")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # ── 3. OpenCV pencereleri + trackbar ────────────────────────────────────
    params = DEFAULT_PARAMS.copy()
    setup_trackbars(params)

    cv2.namedWindow(MAIN_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(MAIN_WIN, RESIZE_W * 2 + 4, RESIZE_H)

    print("Sistem devrede. Çıkmak için 'q' tuşuna basın.")

    fps_frames = 0
    fps_t0     = time.perf_counter()
    fps_disp   = 0.0
    frame_delay = int(1000 / FPS_LIMIT)

    # ── 4. Ana döngü ────────────────────────────────────────────────────────
    while True:
        t_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str):        # video bitti → başa sar
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("Kameradan görüntü alınamıyor!")
            break

        frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))

        # Trackbar parametrelerini oku
        read_trackbars(params)

        # Görüntü işleme (lane_processor)
        result, left_line, right_line, debug = process_frame(frame, params)

        # FPS hesapla
        fps_frames += 1
        elapsed = time.perf_counter() - fps_t0
        if elapsed >= 1.0:
            fps_disp   = fps_frames / elapsed
            fps_frames = 0
            fps_t0     = time.perf_counter()

        proc_ms = (time.perf_counter() - t_start) * 1000
        cv2.putText(
            result,
            f"FPS: {fps_disp:.1f}  |  {proc_ms:.1f} ms",
            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2,
        )

        # Sol: sonuç  |  Sağ: debug maskesi
        sep      = np.zeros((RESIZE_H, 4, 3), dtype=np.uint8)
        combined = np.hstack([result, sep, debug])
        cv2.imshow(MAIN_WIN, combined)
        draw_tb_info(params)

        # ROS publish (lane_ros_publisher)
        if ros_node:
            ros_node.publish(result, left_line, right_line, fps_disp)
            ros_node.spin_once()

        if cv2.waitKey(frame_delay) & 0xFF == ord("q"):
            break

    # ── 5. Temizlik ─────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    if ros_node:
        ros_node.destroy()


if __name__ == "__main__":
    main()
