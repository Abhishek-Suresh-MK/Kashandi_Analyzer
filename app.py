import io
import os
import base64
import sqlite3
from math import ceil
from datetime import datetime
from flask import Flask, request, jsonify, render_template, g
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

# -------- Configuration --------
DATABASE = "leaderboard.db"
# hair density heuristic parameters (tunable)
SKIN_Y_MIN, SKIN_Y_MAX = 0, 255
SKIN_CR_MIN, SKIN_CR_MAX = 133, 173
SKIN_CB_MIN, SKIN_CB_MAX = 77, 127

app = Flask(__name__)

# ---------------- Mediapipe setup ----------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1,
                             refine_landmarks=False, min_detection_confidence=0.5)

# ---------------- Database helpers ----------------
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

def init_db():
    with app.app_context():
        db = get_db()
        db.execute("""
            CREATE TABLE IF NOT EXISTS leaderboard (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nickname TEXT NOT NULL,
                kashandiness INTEGER NOT NULL,
                hair_density REAL NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        db.commit()

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

# ---------------- Utility: base64 dataURL -> BGR image ----------------
def dataurl_to_bgr(dataurl):
    header, encoded = dataurl.split(",", 1)
    data = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return bgr

# ---------------- Hair density heuristic ----------------
def estimate_hair_density(bgr_image, top_y, eyebrow_y):
    h, w = bgr_image.shape[:2]
    y1 = max(0, int(top_y))
    y2 = min(h-1, int(eyrow_safe(eyebrow_y, h)))
    if y2 <= y1:
        return 0.0

    region = bgr_image[y1:y2, int(0.05*w):int(0.95*w)]
    if region.size == 0:
        return 0.0

    ycrcb = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
    y_channel, cr, cb = cv2.split(ycrcb)

    skin_mask = cv2.inRange(ycrcb,
                            np.array([SKIN_Y_MIN, SKIN_CR_MIN, SKIN_CB_MIN]),
                            np.array([SKIN_Y_MAX, SKIN_CR_MAX, SKIN_CB_MAX]))

    _, dark_mask = cv2.threshold(y_channel, 110, 255, cv2.THRESH_BINARY_INV)
    hair_candidate = cv2.bitwise_and(dark_mask, cv2.bitwise_not(skin_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    hair_clean = cv2.morphologyEx(hair_candidate, cv2.MORPH_OPEN, kernel, iterations=1)
    hair_clean = cv2.morphologyEx(hair_clean, cv2.MORPH_DILATE, kernel, iterations=1)

    hair_pixels = cv2.countNonZero(hair_clean)
    total_pixels = hair_clean.shape[0] * hair_clean.shape[1]
    coverage = (hair_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    return round(max(0.0, min(100.0, coverage)), 2)

def eyrow_safe(eyebrow_y, h):
    try:
        return float(eyebrow_y)
    except:
        return 0.0

# ---------------- Core function: detect landmarks, compute metrics ----------------
def analyze_kashandiness(bgr_image):
    h, w = bgr_image.shape[:2]
    img_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return {"error": "No face detected"}, None

    lm = results.multi_face_landmarks[0].landmark
    pts = np.array([[int(p.x * w), int(p.y * h)] for p in lm])

    top_y = pts[:,1].min()
    bottom_y = pts[:,1].max()
    face_height = bottom_y - top_y if bottom_y > top_y else 1

    idxs = set()
    for conn in mp_face.FACEMESH_LEFT_EYEBROW:
        idxs.add(conn[0]); idxs.add(conn[1])
    for conn in mp_face.FACEMESH_RIGHT_EYEBROW:
        idxs.add(conn[0]); idxs.add(conn[1])
    eyebrow_indices = sorted(list(idxs))

    eyebrow_ys = [pts[i,1] for i in eyebrow_indices if i < len(pts)]
    if not eyebrow_ys:
        eyebrow_y = top_y + 0.18 * face_height
    else:
        eyebrow_y = float(np.mean(eyebrow_ys))

    forehead_height = max(0.0, eyebrow_y - top_y)
    forehead_ratio = forehead_height / face_height

    baseline = 0.18
    scale = 500.0
    raw_pct = (forehead_ratio - baseline) * scale
    kashandiness_pct = int(max(0, min(100, round(raw_pct, 0))))

    confidence = min(0.99, 0.5 + (0.5 * (face_height / h)))

    hair_density = estimate_hair_density(bgr_image, top_y, eyebrow_y)

    annual_loss_rate = 3.0
    years_left = max(0, int(ceil((100 - kashandiness_pct) / annual_loss_rate)))

    annotated = bgr_image.copy()
    cv2.rectangle(annotated, (0, int(top_y)), (w-1, int(bottom_y)), (200,200,200), 1)
    for i in eyebrow_indices:
        if i < len(pts):
            cv2.circle(annotated, tuple(pts[i]), 2, (0,255,0), -1)
    y1 = int(top_y)
    y2 = int(eyebrow_y)
    cv2.rectangle(annotated, (int(0.05*w), y1), (int(0.95*w), y2), (255,0,0), 2)
    cv2.putText(annotated, f"kashandiness: {kashandiness_pct}%", (10, h-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
    cv2.putText(annotated, f"hair density: {hair_density}%", (10, h-60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,255,180), 2)

    _, buf = cv2.imencode('.png', annotated)
    annotated_b64 = base64.b64encode(buf).decode('utf-8')
    annotated_dataurl = f"data:image/png;base64,{annotated_b64}"

    result = {
        "kashandiness_pct": kashandiness_pct,
        "years_left": years_left,
        "confidence": round(float(confidence), 2),
        "forehead_ratio": round(float(forehead_ratio), 3),
        "hair_density": round(float(hair_density), 2)
    }
    return result, annotated_dataurl

# ---------------- Flask routes ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image provided"}), 400
    try:
        bgr = dataurl_to_bgr(data['image'])
    except Exception as e:
        return jsonify({"error": "Invalid image data", "detail": str(e)}), 400

    result, annotated_dataurl = analyze_kashandiness(bgr)
    if 'error' in result:
        return jsonify(result), 200
    return jsonify({"result": result, "annotated": annotated_dataurl}), 200

@app.route('/leaderboard/add', methods=['POST'])
def add_leaderboard():
    payload = request.get_json()
    if not payload:
        return jsonify({"error": "No payload"}), 400
    nickname = payload.get("nickname", "").strip()[:40] or "Anon"
    try:
        kash = int(payload.get("kashandiness", 0))
        hair_density = float(payload.get("hair_density", 0.0))
        confidence = float(payload.get("confidence", 0.0))
    except:
        return jsonify({"error": "Invalid numeric values"}), 400

    db = get_db()
    db.execute("INSERT INTO leaderboard (nickname, kashandiness, hair_density, confidence, created_at) VALUES (?, ?, ?, ?, ?)",
               (nickname, kash, hair_density, confidence, datetime.utcnow().isoformat()))
    db.commit()
    return jsonify({"ok": True}), 200

@app.route('/leaderboard/top', methods=['GET'])
def top_leaderboard():
    db = get_db()
    cur_top = db.execute("SELECT nickname, kashandiness, hair_density, confidence, created_at FROM leaderboard ORDER BY kashandiness DESC, created_at DESC LIMIT 10")
    top = [dict(r) for r in cur_top.fetchall()]
    cur_bot = db.execute("SELECT nickname, kashandiness, hair_density, confidence, created_at FROM leaderboard ORDER BY kashandiness ASC, created_at DESC LIMIT 10")
    bottom = [dict(r) for r in cur_bot.fetchall()]
    return jsonify({"top": top, "bottom": bottom}), 200

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
