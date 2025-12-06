"""
AdvancedDrawingPad - single-file runnable app.

Features:
- MediaPipe hand gestures: index fingertip = pointer, pinch (thumb+index) = press/draw
- AI auto-correction: line / circle / rectangle heuristics
- Layers: add, toggle visibility, merge down
- Brush textures: brush, pencil, highlighter, spray
- Zoom & basic pan via offset
- Save PNG and export PDF
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import os
from PIL import Image

# ---------------- CONFIG ----------------
WIDTH, HEIGHT = 1000, 700
NAV_H = 90
BG_COLOR = (10, 10, 10)
PALETTE = [
    (0,0,255),(0,165,255),(0,255,255),(0,255,0),
    (255,0,0),(255,0,255),(255,255,255),(255,255,0)
]
SAVE_DIR = "drawings_advanced"
os.makedirs(SAVE_DIR, exist_ok=True)

# State
current_color = PALETTE[0]
tool = "brush"  # brush | pencil | highlighter | spray | erase | shape
shape_mode = "auto"  # auto | line | rect | circle | none
thickness = 8
pointer_radius = 6

# Layers
layers = []
def new_layer(name="Layer"):
    canv = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    canv[:] = BG_COLOR
    layers.append({"name": f"{name}_{len(layers)+1}", "canvas": canv, "visible": True})

new_layer("Base")
active_layer = 0

# stroke buffer
stroke = []

# zoom/pan
zoom = 1.0
offset = np.array([0.0, 0.0])
min_zoom, max_zoom = 0.5, 3.0

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# pinch debounce
last_pinched = 0
pinch_cooldown = 0.08

# simple history (undo per-layer not implemented to keep script shorter)
history_stack = []

# ---------------- Utilities ----------------
def save_png(layers_list, filename=None):
    if filename is None:
        filename = os.path.join(SAVE_DIR, f"drawing_{int(time.time())}.png")
    h,w,_ = layers_list[0]['canvas'].shape
    merged = np.zeros((h,w,3), dtype=np.uint8)
    merged[:] = BG_COLOR
    for li in layers_list:
        if li.get('visible', True):
            merged = cv2.addWeighted(merged,1.0, li['canvas'],1.0, 0)
    cv2.imwrite(filename, merged)
    print("Saved:", filename)
    return filename

def export_pdf(png_path, pdf_path=None):
    if pdf_path is None:
        pdf_path = png_path.rsplit('.',1)[0] + '.pdf'
    pil = Image.open(png_path).convert('RGB')
    pil.save(pdf_path, 'PDF', resolution=100.0)
    print("Exported PDF:", pdf_path)
    return pdf_path

def screen_to_canvas(pt):
    x_s, y_s = pt
    x = (x_s - offset[0]) / zoom
    y = (y_s - offset[1]) / zoom
    return int(x), int(y)

def draw_layers(base):
    comp = base.copy()
    for li in layers:
        if li["visible"]:
            comp = cv2.addWeighted(comp, 1.0, li["canvas"], 1.0, 0)
    return comp

# ---------- AI shape detection heuristics ----------
def is_line(points, tol=8):
    if len(points) < 6: return False
    pts = np.array(points)
    x1,y1 = pts[0]; x2,y2 = pts[-1]
    dx = x2 - x1; dy = y2 - y1
    if dx == 0 and dy == 0: return False
    m = dy / (dx + 1e-8)
    num = np.sqrt(m*m + 1)
    d = np.abs((pts[:,0]*m - pts[:,1] - (x1*m - y1)) / num)
    return np.mean(d) < tol

def is_circle(points):
    pts = np.array(points)
    if len(pts) < 8: return False, None
    p0 = pts[0]
    idx = np.argmax(np.sum((pts - p0)**2, axis=1))
    pmax = pts[idx]
    center = ((p0 + pmax) / 2.0)
    r = np.mean(np.linalg.norm(pts - center, axis=1))
    residual = np.abs(np.linalg.norm(pts - center, axis=1) - r)
    if np.mean(residual) < r * 0.25:
        return True, (int(center[0]), int(center[1]), int(r))
    return False, None

def is_rect(points):
    pts = np.array(points)
    if len(pts) < 8: return False, None
    x_min, x_max = np.min(pts[:,0]), np.max(pts[:,0])
    y_min, y_max = np.min(pts[:,1]), np.max(pts[:,1])
    if np.ptp(pts[:,0])>20 and np.ptp(pts[:,1])>20:
        return True, (int(x_min),int(y_min),int(x_max),int(y_max))
    return False, None

# ---------- brush textures ----------
def draw_point(canvas, x, y, color, thickness, texture="brush"):
    if texture == "brush":
        cv2.circle(canvas, (x,y), max(1, thickness), color, -1)
    elif texture == "pencil":
        for i in range(3):
            rx = int(x + np.random.randn()*1.2)
            ry = int(y + np.random.randn()*1.2)
            cv2.circle(canvas,(rx,ry), max(1, thickness//3), color, -1)
    elif texture == "highlighter":
        overlay = canvas.copy()
        cv2.circle(overlay,(x,y), int(thickness*1.6), color, -1)
        cv2.addWeighted(overlay, 0.2, canvas, 0.8, 0, canvas)
    elif texture == "spray":
        radius = max(3, thickness*2)
        for _ in range(15):
            r = int(np.random.rand()*radius)
            ang = np.random.rand()*2*np.pi
            rx = int(x + r*np.cos(ang))
            ry = int(y + r*np.sin(ang))
            if 0 <= rx < canvas.shape[1] and 0 <= ry < canvas.shape[0]:
                cv2.circle(canvas, (rx,ry), max(1, int(thickness*0.5)), color, -1)

# ---------------- Main loop ----------------
cap = cv2.VideoCapture(0)
cv2.namedWindow("AdvancedPad", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AdvancedPad", 1200, 800)

last_zoom_distance = None

print("Starting AdvancedDrawingPad. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not detected. Exiting.")
        break
    frame = cv2.flip(frame, 1)
    ui = np.zeros((NAV_H + HEIGHT, WIDTH, 3), dtype=np.uint8)
    ui[:,:] = BG_COLOR

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    pointer_screen = (WIDTH//2, NAV_H + HEIGHT//2)
    pinch = False

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        lm = hand.landmark
        ix = int(lm[8].x * WIDTH)
        iy = int(lm[8].y * (NAV_H + HEIGHT))
        pointer_screen = (ix, iy)
        tx = int(lm[4].x * WIDTH)
        ty = int(lm[4].y * (NAV_H + HEIGHT))
        dist = np.hypot(ix - tx, iy - ty)
        if dist < 40:
            if time.time() - last_pinched > pinch_cooldown:
                pinch = True
                last_pinched = time.time()

        # two-finger zoom (index + middle)
        mx = int(lm[12].x * WIDTH)
        my = int(lm[12].y * (NAV_H + HEIGHT))
        two_dist = np.hypot(ix - mx, iy - my)
        if two_dist > 50:
            if last_zoom_distance is None:
                last_zoom_distance = two_dist
            else:
                change = (two_dist - last_zoom_distance) * 0.005
                zoom = max(min_zoom, min(max_zoom, zoom + change))
                last_zoom_distance = two_dist
        else:
            last_zoom_distance = None

    # draw top palette
    for i,c in enumerate(PALETTE):
        cx = 30 + i*50
        cv2.circle(ui, (cx, 35), 18, c, -1)
        if c == current_color:
            cv2.circle(ui, (cx,35), 22, (255,255,255), 2)

    cv2.putText(ui, f"Tool:{tool} Shape:{shape_mode} Layer:{active_layer+1}/{len(layers)} Zoom:{zoom:.2f}",
                (300,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 2)

    # draw merged layers to display area
    base_canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8); base_canvas[:] = BG_COLOR
    display_canvas = draw_layers(base_canvas)

    # scale for zoom
    disp_scaled = cv2.resize(display_canvas, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
    h_s, w_s = disp_scaled.shape[:2]
    target = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    tx0 = int(np.clip(offset[0], -w_s+10, WIDTH-10))
    ty0 = int(np.clip(offset[1], -h_s+10, HEIGHT-10))
    x0 = tx0; y0 = ty0
    xs = max(0, x0); ys = max(0, y0)
    xe = min(WIDTH, x0 + w_s); ye = min(HEIGHT, y0 + h_s)
    sx = xs - x0; sy = ys - y0
    target[ys:ye, xs:xe] = disp_scaled[sy:sy+ye-ys, sx:sx+xe-xs]

    ui[NAV_H: NAV_H+HEIGHT, :WIDTH] = target

    # draw pointer on UI
    cv2.circle(ui, pointer_screen, int(pointer_radius * zoom), current_color, -1)

    canvas_pt = screen_to_canvas(pointer_screen)

    # pinch interactions
    if pinch:
        # top bar interactions
        if pointer_screen[1] < NAV_H:
            for i,c in enumerate(PALETTE):
                cx = 30 + i*50
                if abs(pointer_screen[0] - cx) < 22:
                    current_color = c
            # left-top hotspots
            if pointer_screen[0] < 80 and pointer_screen[1] < 60:
                new_layer("Layer"); active_layer = len(layers)-1
                time.sleep(0.08)
            if 80 < pointer_screen[0] < 160 and pointer_screen[1] < 60:
                layers[active_layer]["visible"] = not layers[active_layer]["visible"]
                time.sleep(0.08)
            if 160 < pointer_screen[0] < 240 and pointer_screen[1] < 60:
                if active_layer > 0:
                    layers[active_layer-1]["canvas"] = cv2.addWeighted(layers[active_layer-1]["canvas"],1.0,layers[active_layer]["canvas"],1.0,0)
                    layers.pop(active_layer); active_layer = max(0, active_layer-1)
                    time.sleep(0.08)
            if 240 < pointer_screen[0] < 320 and pointer_screen[1] < 60:
                save_png(layers)
                time.sleep(0.08)
            if 320 < pointer_screen[0] < 400 and pointer_screen[1] < 60:
                png = save_png(layers); export_pdf(png); time.sleep(0.08)
        else:
            lx = layers[active_layer]["canvas"]
            x_c, y_c = canvas_pt
            stroke.append((x_c,y_c))
            if tool == "erase":
                cv2.circle(lx, (x_c,y_c), max(4, int(thickness * zoom)), BG_COLOR, -1)
            else:
                tex = "brush"
                if tool == "pencil": tex = "pencil"
                elif tool == "highlighter": tex = "highlighter"
                elif tool == "spray": tex = "spray"
                draw_point(lx, x_c, y_c, current_color, thickness, texture=tex)

    # on pinch release: attempt shape correction
    if (not pinch) and stroke:
        corrected = False
        if shape_mode in ("auto","line") and is_line(stroke):
            x1,y1 = stroke[0]; x2,y2 = stroke[-1]
            cv2.line(layers[active_layer]["canvas"], (int(x1),int(y1)), (int(x2),int(y2)), current_color, thickness)
            corrected = True
        elif shape_mode in ("auto","circle"):
            ok, params = is_circle(stroke)
            if ok:
                cx,cy,r = params
                cv2.circle(layers[active_layer]["canvas"], (cx,cy), r, current_color, thickness)
                corrected = True
        if not corrected:
            rect_ok, rect = is_rect(stroke)
            if rect_ok:
                x1,y1,x2,y2 = rect
                cv2.rectangle(layers[active_layer]["canvas"], (x1,y1), (x2,y2), current_color, thickness)
                corrected = True
        stroke = []

    # layer badges
    for i, ly in enumerate(layers):
        v = "V" if ly["visible"] else "H"
        cv2.putText(ui, f"{i+1}:{ly['name']}[{v}]", (10, NAV_H + 20 + i*18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

    cv2.imshow("AdvancedPad", ui)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == ord('='):
        zoom = min(max_zoom, zoom + 0.1)
    elif k == ord('-'):
        zoom = max(min_zoom, zoom - 0.1)
    elif k == ord('['):
        active_layer = max(0, active_layer-1)
    elif k == ord(']'):
        active_layer = min(len(layers)-1, active_layer+1)
    elif k == ord('s'):
        save_png(layers)
    elif k == ord('p'):
        png = save_png(layers); export_pdf(png)
    elif k == ord('c'):
        layers[active_layer]["canvas"][:] = BG_COLOR

cap.release()
cv2.destroyAllWindows()
