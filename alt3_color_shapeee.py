"""
Alternatif 3 — Gelismis Serit Tespiti: Renk + Sekil (Gradyan) Birlesimi
+ Bird's Eye View + Sliding Window + Canli Trackbar Ayarlari + ROS 2 Publish Sistemi (opsiyonel)

Yontem:
  1. Renk Filtresi (HLS): Beyaz ve sari piksel maskeleri
  2. Sekil Filtresi (Sobel X Gradyan): Serit kenarlarinin dik gradyanlari
  3. Birlesim (VEYA / VE): Her iki maskeyi birlestirerek daha saglikli binary
  4. Perspektif Donusumu → Kus Bakisi (Bird's Eye View)
  5. Kayan Pencere + 2. derece polinom uyumu

Renk tespiti tek basina golge ve yansimalarda yanilabilir.
Sekil tespiti tek basina asfalt ciziklerini de yakalar.
VEYA birlesimi ikisinin avantajlarini bir araya getirir.
VE birlesimi en katı filtredir, gurultu azdir ama kayiplar olabilir.

Sol panel : Tespit sonucu
Sag panel : Maske gorsellestirmesi
             - Yesil  = sadece renk maskesinden gelen pikseller
             - Kirmizi = sadece gradyan maskesinden gelen pikseller
             - Beyaz  = her iki maskeyi de gecen pikseller (guvenceli)

Kullanim:
  python alt3_color_shape.py
  python alt3_color_shape.py 1
  python alt3_color_shape.py video.mp4
"""

import sys
import json
import time
import cv2
import numpy as np
from typing import Any as _Any


def open_camera(source) -> cv2.VideoCapture:
    """
    Kamera kaynagini acar. Tam ekrani kaplayan duzenli nokta deseni
    (IR projektor paterni) goruluyorsa, bu kodun degil, YANLIS KAMERA
    SECILMESININ belirtisidir: Windows'ta Hello yuz girisi icin kullanilan
    kizilotesi (IR) derinlik kamerasi, indeks 0'da RGB kameranin onune
    gecebilir. cv2.CAP_DSHOW backend'i genelde RGB kamerayi once listeler;
    bu yuzden int kaynaklarda once DSHOW denenir, basarisiz olursa
    varsayilana (MSMF) dusulur.
    """
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(source)

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

CTRL_WIN = "Alt3 - Ayarlar (Renk + Sekil)"
VIEW_WIN = "Alt3 | TESPIT  <-->  RENK(yesil) | GRADYAN(kirmizi) | IKISI(beyaz)"

# Sobel kernel boyutlari (OpenCV icin tek sayi, 1-31 arasi)
KSIZE_MAP = [3, 5, 7, 9, 11, 15]


# ── Trackbar Kontrol Paneli ───────────────────────────────────────────────────

def create_controls() -> None:
    cv2.namedWindow(CTRL_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CTRL_WIN, 480, 580)
    noop = lambda _: None

    # ---- Renk Filtresi -------------------------------------------------------
    cv2.createTrackbar("Beyaz L Min",        CTRL_WIN, 184, 255, noop)
    cv2.createTrackbar("Sari H Min",         CTRL_WIN,  30,  50, noop)
    cv2.createTrackbar("Sari H Max",         CTRL_WIN,  45,  50, noop)

    # ---- Sobel X Gradyan Filtresi --------------------------------------------
    cv2.createTrackbar("Sobel Esik Alt",     CTRL_WIN,  25, 255, noop)
    cv2.createTrackbar("Sobel Esik Ust",     CTRL_WIN, 100, 255, noop)
    # 0..5 → KSIZE_MAP listesindeki kernel boyutlari
    cv2.createTrackbar("Sobel Kernel(0-5)",  CTRL_WIN,   1,   5, noop)

    # ---- Birlesim Modu -------------------------------------------------------
    # 0 = VEYA (OR)  → renk VEYA gradyan → genis kapsam
    # 1 = VE  (AND)  → renk VE  gradyan → guvenceli, daha katı
    cv2.createTrackbar("Mod: 0=VEYA 1=VE",  CTRL_WIN,   1,   1, noop)

    # ---- Perspektif Donusumu -------------------------------------------------
    cv2.createTrackbar("Persp Ust Y %",      CTRL_WIN,  47,  90, noop)
    cv2.createTrackbar("Persp Ust Yaricap%", CTRL_WIN,   13,  25, noop)
    cv2.createTrackbar("Persp Alt Kenar %",  CTRL_WIN,  18,  40, noop)
    cv2.createTrackbar("Persp Alt Y %",      CTRL_WIN,  79, 100, noop)

    # ---- Kayan Pencere -------------------------------------------------------
    cv2.createTrackbar("Pencere Sayisi",     CTRL_WIN,   9,  20, noop)
    cv2.createTrackbar("Pencere Margin",     CTRL_WIN,  80, 200, noop)
    cv2.createTrackbar("Min Piksel",         CTRL_WIN,  60, 200, noop)


def read_params() -> dict:
    g     = lambda k: cv2.getTrackbarPos(k, CTRL_WIN)
    h_min = g("Sari H Min")
    h_max = g("Sari H Max")
    s_low = g("Sobel Esik Alt")
    s_hi  = g("Sobel Esik Ust")
    ust_y = max(1, g("Persp Ust Y %")) / 100.0
    return {
        "white_l_min":  g("Beyaz L Min"),
        "yel_h_min":    h_min,
        "yel_h_max":    max(h_min + 1, h_max),
        "sobel_low":    s_low,
        "sobel_high":   max(s_low + 1, s_hi),
        "ksize":        KSIZE_MAP[g("Sobel Kernel(0-5)")],
        "combine_or":   g("Mod: 0=VEYA 1=VE") == 0,   # True=OR, False=AND
        "ust_y":        ust_y,
        "ust_hw":       max(1, g("Persp Ust Yaricap%")) / 100.0,
        "alt_off":      max(1, g("Persp Alt Kenar %"))  / 100.0,
        "alt_y":        max(ust_y + 0.01, g("Persp Alt Y %") / 100.0),
        "n_windows":    max(2, g("Pencere Sayisi")),
        "margin":       max(10, g("Pencere Margin")),
        "min_pix":      max(5,  g("Min Piksel")),
    }


# ── Ikili Maske Üretimi ───────────────────────────────────────────────────────

def color_binary(frame: np.ndarray, p: dict) -> np.ndarray:
    """HLS uzayinda beyaz + sari serit piksellerini 0/255 binary olarak dondurur."""
    hls    = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white  = cv2.inRange(hls,
                         np.array([0,          p["white_l_min"], 0],   np.uint8),
                         np.array([255,        255,              255],  np.uint8))
    yellow = cv2.inRange(hls,
                         np.array([p["yel_h_min"], 38,  115], np.uint8),
                         np.array([p["yel_h_max"], 204, 255], np.uint8))
    return cv2.bitwise_or(white, yellow)


def gradient_binary(frame: np.ndarray, p: dict) -> np.ndarray:
    """
    Sobel X gradyanini hesaplayarak dikey kenarlari (serit sinirlarini) bulur.
    Yalnizca yatay eksende gradyan alinir; bu serit kenarlarina diktir.
    """
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=p["ksize"])
    abs_x  = np.absolute(sobelx)
    mx     = abs_x.max()
    if mx == 0:
        return np.zeros_like(gray)
    scaled = np.uint8(255.0 * abs_x / mx)
    result = np.zeros_like(scaled)
    result[(scaled >= p["sobel_low"]) & (scaled <= p["sobel_high"])] = 255
    return result


def combined_binary(frame: np.ndarray, p: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Renk ve gradyan maskelerini birlestirerek tek bir binary dondurur.
    Ayrica gorsellestirme icin her maskeyi ayri dondurur.
    """
    c_bin = color_binary(frame, p)
    g_bin = gradient_binary(frame, p)

    if p["combine_or"]:
        combined = cv2.bitwise_or(c_bin, g_bin)
    else:
        combined = cv2.bitwise_and(c_bin, g_bin)

    return combined, c_bin, g_bin


def mask_visualization(c_bin: np.ndarray, g_bin: np.ndarray) -> np.ndarray:
    """
    Yesil  = sadece renk maskesi
    Kirmizi = sadece gradyan maskesi
    Beyaz  = her ikisi (en guvenceli pikseller)
    """
    vis    = np.zeros((c_bin.shape[0], c_bin.shape[1], 3), dtype=np.uint8)
    c_mask = c_bin == 255
    g_mask = g_bin == 255

    vis[c_mask & ~g_mask] = (0,   220,   0)    # sadece renk → yesil
    vis[~c_mask & g_mask] = (0,    60, 200)    # sadece gradyan → kirmizi
    vis[c_mask  & g_mask] = (255, 255, 255)    # her ikisi → beyaz
    return vis


# ── Perspektif Donusumu ───────────────────────────────────────────────────────

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


# ── Kayan Pencere ─────────────────────────────────────────────────────────────

def sliding_window_fit(binary_warped: np.ndarray, p: dict, build_viz: bool = False):
    """
    build_viz=False (varsayilan): debug gorsellestirmesi insa edilmez.
    main()/_cb() zaten viz'i `_` ile atiyordu ama her karede pencere
    dikdortgenleri + polyline cizimi gereksiz yere hesaplaniyordu --
    FPS dususunun buyuk kismi buydu.
    """
    h, w    = binary_warped.shape
    hist    = np.sum(binary_warped[h // 2:, :], axis=0)
    mid     = w // 2
    lx_cur  = int(np.argmax(hist[:mid]))
    rx_cur  = int(np.argmax(hist[mid:])) + mid

    n_win   = p["n_windows"]
    margin  = p["margin"]
    min_pix = p["min_pix"]
    win_h   = h // n_win

    nz_y, nz_x            = binary_warped.nonzero()
    left_inds, right_inds = [], []
    viz = np.dstack([binary_warped] * 3).astype(np.uint8) * 255 if build_viz else None

    for i in range(n_win):
        y_lo = h - (i + 1) * win_h
        y_hi = h - i * win_h
        xl_lo, xl_hi = lx_cur - margin, lx_cur + margin
        xr_lo, xr_hi = rx_cur - margin, rx_cur + margin

        if build_viz:
            assert viz is not None
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

    if build_viz:
        assert viz is not None
        viz[ly, lx] = [255, 80,  80]
        viz[ry, rx] = [80,  80, 255]

    left_fit  = np.polyfit(ly, lx, 2) if len(lx) >= 100 else None
    right_fit = np.polyfit(ry, rx, 2) if len(rx) >= 100 else None

    if build_viz:
        assert viz is not None
        if left_fit is not None:
            ploty = np.linspace(0, h - 1, h)
            pts   = np.column_stack([np.polyval(left_fit, ploty), ploty]).astype(np.int32)
            cv2.polylines(viz, [pts.reshape(-1, 1, 2)], False, (255, 255, 0), 3)
        if right_fit is not None:
            ploty = np.linspace(0, h - 1, h)
            pts   = np.column_stack([np.polyval(right_fit, ploty), ploty]).astype(np.int32)
            cv2.polylines(viz, [pts.reshape(-1, 1, 2)], False, (0, 255, 255), 3)

    return left_fit, right_fit, viz


# ── Lane Overlay + Merkez Yol Cizgisi ────────────────────────────────────────

def draw_lane_overlay(original: np.ndarray, binary_warped: np.ndarray,
                      left_fit, right_fit,
                      Minv: np.ndarray) -> tuple[np.ndarray, float | None]:
    """
    Serit alani, sol/sag cizgi ve kirmizi kesik merkez rehber cizgisini cizer.
    offset: serit merkezi - kare merkezi (piksel); pozitif=saga, negatif=sola.
    """
    if left_fit is None or right_fit is None:
        return original, None

    h, w   = binary_warped.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    ploty  = np.linspace(0, h - 1, h)

    lx = np.polyval(left_fit,  ploty)
    rx = np.polyval(right_fit, ploty)
    cx = (lx + rx) / 2.0                    # merkez yol noktaları

    pts_l = np.column_stack([lx, ploty]).astype(np.int32).reshape(-1, 1, 2)
    pts_r = np.column_stack([rx, ploty]).astype(np.int32).reshape(-1, 1, 2)[::-1]

    # Serit alani dolgusu (yesil)
    cv2.fillPoly(canvas, [np.concatenate([pts_l, pts_r])], (0, 160, 0))
    # Sol serit siniri (mavi), sag serit siniri (kirmizi)
    cv2.polylines(canvas, [pts_l],       False, (255, 0,   0), 8)
    cv2.polylines(canvas, [pts_r[::-1]], False, (0,   0, 200), 8)

    # ── Kirmizi Kesik Merkez Yol Cizgisi ─────────────────────────────────────
    pts_c    = np.column_stack([cx, ploty]).astype(np.int32)
    dash_on  = 22    # cizgi uzunlugu (piksel cinsinden y)
    dash_off = 16    # bosluk uzunlugu
    i = 0
    while i + dash_on < len(pts_c):
        p0 = tuple(pts_c[i])
        p1 = tuple(pts_c[min(i + dash_on, len(pts_c) - 1)])
        cv2.line(canvas, p0, p1, (0, 30, 255), 5)
        i += dash_on + dash_off

    unwarped = cv2.warpPerspective(canvas, Minv, (original.shape[1], original.shape[0]))
    result   = cv2.addWeighted(original, 1.0, unwarped, 0.38, 0)

    # Offset: kus bakisinda serit merkezi ile kare merkezi arasindaki mesafe
    lane_cx = (np.polyval(left_fit, h - 1) + np.polyval(right_fit, h - 1)) / 2.0
    offset  = lane_cx - w / 2.0   # + = serit merkezi sagda → sola don

    return result, offset


def draw_steering_indicator(img: np.ndarray, offset: float | None) -> None:
    """
    Sol panel altina aracin serit merkezinden sapmasini gosteren bar ve yazi cizer.
    Sari daire = aracin anlık konumu.
    Yesil kutu  = guvenli merkez bolgesi (±20 px tolerans).
    """
    h, w = img.shape[:2]

    bar_w = 320
    bar_h = 18
    bar_x = (w - bar_w) // 2
    bar_y = h - 42
    mid   = bar_x + bar_w // 2

    # Arka plan ve yesil guvenli bolge
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
    cv2.rectangle(img, (mid - 32, bar_y), (mid + 32, bar_y + bar_h), (0, 110, 0), -1)
    cv2.line(img, (mid, bar_y - 2), (mid, bar_y + bar_h + 2), (200, 200, 200), 1)

    # Sol / Sag etiketleri
    cv2.putText(img, "SOL", (bar_x + 2, bar_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    cv2.putText(img, "SAG", (bar_x + bar_w - 30, bar_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    if offset is None:
        return

    # Aracin konumunu gosteren sari daire
    scale    = 0.45                                        # piksel → bar piksel
    marker_x = int(np.clip(mid + offset * scale, bar_x + 8, bar_x + bar_w - 8))
    cv2.circle(img, (marker_x, bar_y + bar_h // 2), 9, (0, 220, 255), -1)

    # Yon yazisi
    if abs(offset) < 20:
        txt = "MERKEZ - DUZGUN GIT"
        col = (0, 255, 100)
    elif offset > 0:
        txt = f"<< SOL'A DON  ({offset:+.0f}px)"
        col = (0, 140, 255)
    else:
        txt = f"SAG'A DON >>  ({offset:+.0f}px)"
        col = (0, 140, 255)

    tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0]
    cv2.putText(img, txt, ((w - tw) // 2, bar_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)


# ── Yan Yana Birlestirme ─────────────────────────────────────────────────────

def make_combined(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    for img, label in [(left, " TESPIT "), (right, " MASKE ANALIZI ")]:
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

    cap = open_camera(source)
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
    print("[BILGI] Sag panelde: Yesil=Renk  Kirmizi=Gradyan  Beyaz=Her ikisi")
    print("[BILGI] Eger goruntude duzenli nokta/benek deseni goruyorsaniz bu "
          "kodla ilgili degildir -- IR (kizilotesi) kamera secilmis olabilir. "
          "'python alt3_color_shape.py 1' (veya 2, 3...) ile farkli bir "
          "kamera indeksi deneyin.")

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        frame = cv2.resize(frame, (PROC_W, PROC_H))
        p     = read_params()

        # ── Maske Uretimi ─────────────────────────────────────────────────────
        comb_bin, c_bin, g_bin = combined_binary(frame, p)

        # ── Perspektif + Kayan Pencere ─────────────────────────────────────────
        M, Minv, src_pts = build_transform(PROC_H, PROC_W, p)
        warped           = cv2.warpPerspective(comb_bin, M, (PROC_W, PROC_H))
        left_fit, right_fit, _ = sliding_window_fit(warped, p)

        # ── Sol Panel: Tespit Sonucu ───────────────────────────────────────────
        left_panel, offset = draw_lane_overlay(
            frame.copy(), warped, left_fit, right_fit, Minv
        )

        # Perspektif trapezi
        ordered = src_pts[[0, 1, 3, 2]].astype(np.int32)
        cv2.polylines(left_panel, [ordered.reshape(-1, 1, 2)], True, (0, 255, 255), 1)

        found      = left_fit is not None and right_fit is not None
        status_col = (0, 255, 0) if found else (0, 0, 255)
        mode_str   = "VEYA (OR)" if p["combine_or"] else "VE  (AND)"

        info = [
            f"Renk: beyaz_L>={p['white_l_min']}  sari_H={p['yel_h_min']}-{p['yel_h_max']}",
            f"Gradyan: esik={p['sobel_low']}-{p['sobel_high']}  kernel={p['ksize']}",
            f"Birlesim modu: {mode_str}",
            f"Persp: ustY={int(p['ust_y']*100)}%  hw={int(p['ust_hw']*100)}%"
            f"  altK={int(p['alt_off']*100)}%  altY={int(p['alt_y']*100)}%",
            f"Win: sayi={p['n_windows']}  margin={p['margin']}  min={p['min_pix']}",
        ]
        for i, txt in enumerate(info):
            cv2.putText(left_panel, txt, (8, 26 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)
        cv2.putText(left_panel,
                    "SERIT BULUNDU" if found else "SERIT YOK",
                    (8, PROC_H - 68), cv2.FONT_HERSHEY_SIMPLEX, 0.70, status_col, 2)

        # Yonlendirme gostergesi (bar + yazi)
        draw_steering_indicator(left_panel, offset)

        # ── Sag Panel: Renk-Kodlu Maske Gorsellestirmesi ───────────────────────
        right_panel = mask_visualization(c_bin, g_bin)

        # Maskenin kus bakisi halini alt yarisinda goster
        # (Eskiden mask_visualization() burada IKINCI KEZ ayni girdiyle
        #  cagiriliyordu -- gereksiz tekrar hesaplama FPS'i dusuruyordu.)
        warped_vis = cv2.warpPerspective(right_panel, M, (PROC_W, PROC_H))
        right_panel[PROC_H // 2:, :] = warped_vis[PROC_H // 2:, :]

        # Ust/alt ayrimi cizgisi
        cv2.line(right_panel, (0, PROC_H // 2), (PROC_W, PROC_H // 2),
                 (200, 200, 200), 1)
        cv2.putText(right_panel, "ORIJINAL MASKE",
                    (8, PROC_H // 2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(right_panel, "KUS BAKISI MASKE",
                    (8, PROC_H // 2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Maske legend
        for i, (col, txt) in enumerate([
            ((0, 220, 0),   "Yesil  = Renk maskesi"),
            ((0, 60, 200),  "Kirmizi= Gradyan maskesi"),
            ((255,255,255), "Beyaz  = Her ikisi"),
        ]):
            cv2.putText(right_panel, txt,
                        (8, PROC_H - 60 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)

        cv2.imshow(VIEW_WIN, make_combined(left_panel, right_panel))

        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── ROS 2 Publish Sistemi ────────────────────────────────────────────────────

DEFAULT_PARAMS: dict = {
    "white_l_min": 184, "yel_h_min": 30,  "yel_h_max": 45,
    "sobel_low":    25, "sobel_high": 100, "ksize": 5,
    "combine_or": False,
    "ust_y":  0.47, "ust_hw": 0.13, "alt_off": 0.18, "alt_y": 0.79,
    "n_windows": 9, "margin": 80,   "min_pix": 60,
}


class LanePublisherNode(_RosNode):  # type: ignore[misc]
    """
    Alt-3 Renk+Sekil (Sobel) → ROS 2 publisher.

    Topics:
      /lane/processed_image             sensor_msgs/Image
      /lane/processed_image/compressed  sensor_msgs/CompressedImage
      /lane/data                        std_msgs/String  (JSON)

    JSON veri ornegi:
      { "algorithm": "color_shape", "combine_mode": "AND",
        "left_fit": [...], "right_fit": [...],
        "offset_px": 12.5, "lanes_detected": true, "fps": 24.8 }

    Calistirma:
      python alt3_color_shape.py --ros [--source video.mp4] [--fps 25] [--no-gui]
    """

    def __init__(self, source, fps_limit: int = 25,
                 jpeg_quality: int = 75, show_local: bool = True) -> None:
        super().__init__("lane_publisher_alt3")
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

        self.cap = open_camera(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Video kaynagi acilamadi: {source}")

        if show_local:
            create_controls()
            cv2.namedWindow(VIEW_WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(VIEW_WIN, 1600, 450)

        self.timer = self.create_timer(1.0 / fps_limit, self._cb)
        self.get_logger().info(
            f"[Alt3-ColorShape] src={source} | {fps_limit}fps | yerel={show_local}"
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

        comb_bin, c_bin, g_bin           = combined_binary(frame, p)
        M, Minv, src_pts                 = build_transform(PROC_H, PROC_W, p)
        warped                           = cv2.warpPerspective(comb_bin, M, (PROC_W, PROC_H))
        left_fit, right_fit, _           = sliding_window_fit(warped, p)
        left_panel, offset               = draw_lane_overlay(frame.copy(), warped, left_fit, right_fit, Minv)

        ordered = src_pts[[0, 1, 3, 2]].astype(np.int32)
        cv2.polylines(left_panel, [ordered.reshape(-1, 1, 2)], True, (0, 255, 255), 1)

        found        = left_fit is not None and right_fit is not None
        combine_mode = "OR" if p["combine_or"] else "AND"

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
            right_panel = mask_visualization(c_bin, g_bin)
            cv2.imshow(VIEW_WIN, make_combined(left_panel, right_panel))
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
            "algorithm":      "color_shape",
            "combine_mode":   combine_mode,
            "left_fit":       left_fit.tolist()  if left_fit  is not None else None,
            "right_fit":      right_fit.tolist() if right_fit is not None else None,
            "offset_px":      round(offset, 2)   if offset    is not None else None,
            "lanes_detected": found,
            "fps":            round(self._fps_disp, 1),
        })))

    def destroy_node(self) -> None:
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main_ros() -> None:
    """ROS 2 modunda baslatir.  python alt3_color_shape.py --ros [...]"""
    if not _HAS_ROS:
        print("[HATA] rclpy / cv_bridge yuklu degil.")
        print("  Standart mod: python alt3_color_shape.py [kaynak]")
        return

    import argparse
    ap = argparse.ArgumentParser(prog="alt3_color_shape (ROS modu)")
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
