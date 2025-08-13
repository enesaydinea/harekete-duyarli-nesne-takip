#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Harekete Duyarlı Nesne Takip Projesi
------------------------------------
- Kaynak: Webcam (varsayılan) veya video dosyası (--source)
- Hareket algılama: BackgroundSubtractorMOG2
- Nesne çıkarımı: Morfolojik işlemler + kontur/alan filtresi
- Takip: Basit centroid tabanlı eşleme (yakınlık ile)
- ID ve renk ataması: Kalıcı, ID'den türetilmiş rastgele renk
- HUD: FPS (üstel ortalama), nesne sayısı, tuş ipuçları
- Kapanış: 'q' / ESC ile veya pencere 'X' butonu
- Kısayollar:
    m : Maske görünümünü aç/kapat
    p : Pause/Resume
    r : Arkaplan modelini sıfırla
    s : Ekran görüntüsü kaydet
    q : Çıkış (ESC de çalışır)

Gereksinimler:
    pip install opencv-python numpy

Örnek:
    python motion_track.py --source 0
    python motion_track.py --source sample.mp4
"""

import cv2
import numpy as np
import time
import argparse
from collections import deque

# --------------- Yardımcı fonksiyonlar ---------------

def seeded_color_for_id(id_int: int):
    """ID'den deterministik BGR renk üretir (canlı, doygun)."""
    rng = np.random.default_rng(id_int * 987654321 + 12345)
    h = rng.integers(0, 180)      # OpenCV HSV hue: [0,180)
    s = 200 + rng.integers(0, 56) # [200,255]
    v = 200 + rng.integers(0, 56)
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0].tolist()
    return tuple(int(x) for x in bgr)

def get_centroid(box):
    x, y, w, h = box
    return (int(x + w/2), int(y + h/2))

# --------------- Basit Centroid Tracker ---------------

class CentroidTracker:
    def __init__(self, max_distance=60, max_lost=15):
        self.next_id = 1
        self.objects = dict()      # id -> (box, centroid)
        self.lost_counts = dict()  # id -> frames since last seen
        self.trails = dict()       # id -> deque of points for trail effect
        self.max_distance = max_distance
        self.max_lost = max_lost

    def update(self, detections):
        # detections: list of (x,y,w,h)
        det_centroids = [get_centroid(b) for b in detections]

        # Eğer henüz hiçbir obje yoksa, hepsini kaydet
        if len(self.objects) == 0:
            for b, c in zip(detections, det_centroids):
                oid = self.next_id
                self.next_id += 1
                self.objects[oid] = (b, c)
                self.lost_counts[oid] = 0
                self.trails[oid] = deque(maxlen=32)
                self.trails[oid].append(c)
            return self.objects

        # Mevcut objeleri ve centroidlerini listeler
        obj_ids = list(self.objects.keys())
        obj_centroids = [self.objects[i][1] for i in obj_ids]

        # Eşleme: basit en yakın centroid (greedy), maksimum mesafe eşiği ile
        unmatched_dets = set(range(len(detections)))
        unmatched_objs = set(obj_ids)

        # Distance matrix hesapla
        dists = np.zeros((len(obj_centroids), len(det_centroids)), dtype=np.float32)
        for i, oc in enumerate(obj_centroids):
            for j, dc in enumerate(det_centroids):
                dists[i, j] = np.linalg.norm(np.array(oc) - np.array(dc))

        # Greedy eşleme (en küçük mesafeden başlayarak)
        if dists.size > 0:
            pairs = []
            coords = [(i, j) for i in range(dists.shape[0]) for j in range(dists.shape[1])]
            coords.sort(key=lambda x: dists[x[0], x[1]])
            used_objs = set()
            used_dets = set()
            for i, j in coords:
                if i in used_objs or j in used_dets:
                    continue
                if dists[i, j] <= self.max_distance:
                    pairs.append((i, j))
                    used_objs.add(i)
                    used_dets.add(j)

            # Eşlenenleri güncelle
            for i, j in pairs:
                oid = obj_ids[i]
                b = detections[j]
                c = det_centroids[j]
                self.objects[oid] = (b, c)
                self.lost_counts[oid] = 0
                self.trails[oid].append(c)
                if j in unmatched_dets:
                    unmatched_dets.remove(j)
                if oid in unmatched_objs:
                    unmatched_objs.remove(oid)

        # Eşlenmeyen yeni detections için yeni ID aç
        for j in list(unmatched_dets):
            b = detections[j]
            c = det_centroids[j]
            oid = self.next_id
            self.next_id += 1
            self.objects[oid] = (b, c)
            self.lost_counts[oid] = 0
            self.trails[oid] = deque(maxlen=32)
            self.trails[oid].append(c)

        # Eşlenmeyen objeler için lost++ ve gerekiyorsa sil
        to_delete = []
        for oid in list(unmatched_objs):
            self.lost_counts[oid] += 1
            if self.lost_counts[oid] > self.max_lost:
                to_delete.append(oid)
        for oid in to_delete:
            self.objects.pop(oid, None)
            self.lost_counts.pop(oid, None)
            self.trails.pop(oid, None)

        return self.objects

# --------------- Ana Uygulama ---------------

def main():
    parser = argparse.ArgumentParser(description="Harekete Duyarlı Nesne Takip Projesi")
    parser.add_argument("--source", type=str, default="0", help="0 (webcam) veya video dosya yolu")
    parser.add_argument("--min-area", type=int, default=800, help="Kontur alanı alt sınırı (px)")
    parser.add_argument("--max-dist", type=int, default=80, help="Takipte maksimum centroid mesafesi (px)")
    parser.add_argument("--max-lost", type=int, default=15, help="ID kaybetme toleransı (kare)")
    parser.add_argument("--show", type=str, default="frame", choices=["frame", "mask"], help="Başlangıç görünümü")
    args = parser.parse_args()

    # Kaynak
    src = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Kaynak açılamadı:", args.source)
        return

    # Arkaplan çıkarıcı
    backsub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=32, detectShadows=True)

    # Tracker
    tracker = CentroidTracker(max_distance=args.max_dist, max_lost=args.max_lost)

    # FPS ölçümü (üstel ortalama)
    fps_ema = None
    alpha = 0.1  # smoothing faktörü

    # Görünüm ve kontrol
    show_mask = (args.show == "mask")
    paused = False
    snap_idx = 1

    win_name = "Harekete Duyarlı Takip"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    last_time = time.perf_counter()

    while True:
        # Pencere X'e basıldı mı? (getWindowProperty < 1: kapatılıyor)
        prop = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE)
        if prop < 1:
            break

        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Akış bitti veya kare okunamadı.")
                break

            # Gölgeyi zayıflatmak için gri + blur
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5,5), 0)

            # Maske
            fg = backsub.apply(gray)

            # Maske iyileştirme
            # Gölge (127) değerlerini bastır
            _, fg_bin = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
            # Morfoloji
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            fg_bin = cv2.morphologyEx(fg_bin, cv2.MORPH_OPEN, kernel, iterations=2)
            fg_bin = cv2.dilate(fg_bin, kernel, iterations=2)

            # Konturlar
            contours, _ = cv2.findContours(fg_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detections = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < args.min_area:
                    continue
                x,y,w,h = cv2.boundingRect(cnt)
                detections.append((x,y,w,h))

            # Tracker güncelle
            objects = tracker.update(detections)

            # FPS
            now = time.perf_counter()
            dt = now - last_time
            last_time = now
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            fps_ema = inst_fps if fps_ema is None else (alpha * inst_fps + (1 - alpha) * fps_ema)

            # Çizimler
            if show_mask:
                vis = cv2.cvtColor(fg_bin, cv2.COLOR_GRAY2BGR)
            else:
                vis = frame.copy()

            for oid, (box, centroid) in objects.items():
                x,y,w,h = box
                color = seeded_color_for_id(oid)
                cv2.rectangle(vis, (x,y), (x+w, y+h), color, 2)
                cv2.circle(vis, centroid, 3, color, -1)
                label = f"ID {oid}"
                # Arka planlı metin
                (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(vis, (x, y- th - 8), (x + tw + 6, y), (0,0,0), -1)
                cv2.putText(vis, label, (x+3, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)

                # Trail çiz
                trail = tracker.trails.get(oid, [])
                for i in range(1, len(trail)):
                    if trail[i-1] is None or trail[i] is None:
                        continue
                    cv2.line(vis, trail[i-1], trail[i], color, 2)

            # HUD
            info_lines = [
                f"FPS: {fps_ema:.1f}" if fps_ema is not None else "FPS: --",
                f"Nesne: {len(objects)}",
                f"Gorunum: {'Maske' if show_mask else 'Kamera'}",
                "Tuslar: [m] Maske  [p] Pause  [r] Reset BG  [s] Screenshot  [q/ESC] Cikis"
            ]
            for i, txt in enumerate(info_lines):
                cv2.putText(vis, txt, (10, 22 + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        else:
            # Pause modunda sadece HUD metni göster
            vis = np.zeros((480, 640, 3), dtype=np.uint8)
            pause_txt = "PAUSE - Devam icin 'p'"
            cv2.putText(vis, pause_txt, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("Harekete Duyarlı Takip", vis)

        # Klavye
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC or q
            break
        elif key == ord('m'):
            show_mask = not show_mask
        elif key == ord('p'):
            paused = not paused
        elif key == ord('r'):
            backsub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=32, detectShadows=True)
        elif key == ord('s'):
            # Kaydet
            filename = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(filename, vis)
            print("Kaydedildi:", filename)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
