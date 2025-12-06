# AdvancedDrawingPad

# Hand-gesture controlled drawing pad using MediaPipe + OpenCV.

Features:
- Hand gesture control (pinch to draw, two-finger zoom)
- AI shape auto-correction (line, circle, rectangle)
- Layer support (add/toggle/merge)
- Brush textures: brush, pencil, highlighter, spray
- Save PNG and Export PDF
- Undo/Redo basics, Zoom/Pan

# Run:
pip install -r requirements.txt
python main.py

# Controls (gesture + keyboard):
- Pinch (thumb + index close): draw / click
- Two fingers (index + middle) spread: zoom
- Tap top palette to change color
- Tap left-top hotspots for Add Layer / Toggle / Merge / Save / Export
- Keyboard: s = save PNG, p = export PDF, [ and ] = switch layer, Esc = exit
