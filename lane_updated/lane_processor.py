"""
lane_processor.py
Şerit tespiti — saf görüntü işleme, ROS bağımlılığı yok.

Dışa aktarılanlar:
  DEFAULT_PARAMS  : varsayılan parametre sözlüğü
  color_filter()  : HLS renk filtresi (beyaz + sarı)
  apply_roi()     : trapez ROI maskesi
  average_lines() : Hough çizgilerini ortalar
  process_frame() : tam işleme hattı
"""

import cv2
import numpy as np


# ── Varsayılan parametreler ───────────────────────────────────────────────────
DEFAULT_PARAMS: dict = {
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


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

def color_filter(frame: np.ndarray, p: dict) -> np.ndarray:
    """HLS renk uzayında beyaz ve sarı şeritleri maskele."""
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
    """Trapez şeklinde ilgi bölgesi maskesi uygula."""
    mask = np.zeros_like(gray)
    cy   = int(h * p["roi_top"])
    hw   = p["roi_hw"]
    verts = np.array([[
        (int(w * 0.05),        h),
        (int(w * (0.5 - hw)), cy),
        (int(w * (0.5 + hw)), cy),
        (int(w * 0.95),        h),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, verts, 255)
    return cv2.bitwise_and(gray, mask)


def average_lines(frame: np.ndarray, lines, roi_top: float):
    """Hough çizgilerini sol/sağ olarak ayır ve ortala."""
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


# ── Ana işleme fonksiyonu ─────────────────────────────────────────────────────

def process_frame(frame: np.ndarray, p: dict) -> tuple:
    """
    Tek bir frame'i tam işleme hattından geçirir.

    Returns:
        result     : anotasyonlu BGR frame
        left_line  : (x1,y1,x2,y2) veya None
        right_line : (x1,y1,x2,y2) veya None
        debug      : Canny maskesi + ham Hough çizgileri (BGR)
    """
    h, w = frame.shape[:2]

    filtered  = color_filter(frame, p)
    gray      = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blurred   = cv2.GaussianBlur(gray, (5, 5), 0)
    edges     = cv2.Canny(blurred, p["canny_low"], p["canny_high"])
    roi_edges = apply_roi(edges, h, w, p)

    lines = cv2.HoughLinesP(
        roi_edges,
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
    cy     = int(h * p["roi_top"])
    hw     = p["roi_hw"]
    roi_pts = np.array([
        [int(w * 0.05),        h - 1],
        [int(w * (0.5 - hw)), cy],
        [int(w * (0.5 + hw)), cy],
        [int(w * 0.95),        h - 1],
    ], dtype=np.int32)
    cv2.polylines(result, [roi_pts], True, (255, 255, 0), 1)

    # Durum etiketi
    detected = bool(left_line and right_line)
    cv2.putText(
        result,
        "SERIT TESPIT EDILDI" if detected else "SERIT YOK",
        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (0, 255, 0) if detected else (0, 0, 255), 2,
    )

    # ── Debug / Maske karesi ──────────────────────────────────────────────────
    debug = cv2.cvtColor(roi_edges, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(debug, (x1, y1), (x2, y2), (255, 80, 0), 1)

    for coords in (left_line, right_line):
        if coords:
            x1, y1, x2, y2 = coords
            cv2.line(debug, (x1, y1), (x2, y2), (0, 255, 80), 3)

    cv2.polylines(debug, [roi_pts], True, (0, 255, 255), 1)
    cv2.putText(debug, "MASKE + KENAR",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

    return result, left_line, right_line, debug
