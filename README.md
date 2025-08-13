# Harekete Duyarlı Nesne Takip (Staj Projesi)

OpenCV MOG2 arka plan çıkarma + kontur tabanlı nesne tespiti + centroid takip algoritması (ID ve iz/trail).  
Webcam veya video dosyası üzerinden çalışır.

## Kurulum
```bash
# (Opsiyonel) sanal ortam
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
# macOS/Linux:
# source .venv/bin/activate

ÇALIŞTIRMA 
# Webcam ile
python motion_track.py --source 0

# Video dosyası ile
python motion_track.py --source ornek_video.mp4
