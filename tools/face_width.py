import sys, os, cv2, datetime
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QComboBox
)
from PySide6.QtGui import QPixmap, QImage, QTextCursor, QAction
from PySide6.QtCore import Qt

# === Folders ===
script_dir = os.path.dirname(os.path.abspath(__file__))

cropper_output = os.path.join(script_dir, "../images/output_images/cropper")
face_width_input = os.path.join(script_dir, "../images/input_images/face_width")

if os.path.isdir(cropper_output) and not os.path.exists(face_width_input):
    os.symlink(cropper_output, face_width_input)
    print(f"🔗 Symlink created: {face_width_input} → {cropper_output}")
elif os.path.islink(face_width_input):
    print(f"✅ Symlink already exists: {face_width_input} → {os.readlink(face_width_input)}")
elif os.path.isdir(face_width_input):
    print(f"⚠️ Destination exists as a real folder: {face_width_input} — not creating symlink.")
else:
    print(f"⚠️ Cropper output folder missing: {cropper_output}")


print("Symlink points to:", os.readlink(face_width_input))

output_folders = {
    "high": os.path.join(script_dir, "../images/output_images/face_width/0.76_or_more"),
    "mid": os.path.join(script_dir, "../images/output_images/face_width/0.68_to_0.75"),
    "low": os.path.join(script_dir, "../images/output_images/face_width/0_to_0.67"),
}
for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)

progress_file = os.path.join(script_dir, "../progress/face_width.txt")
os.makedirs(os.path.dirname(progress_file), exist_ok=True)

# === Helpers ===
def cvimg_to_qpix(img):
    h, w, ch = img.shape
    bytes_per_line = ch * w
    qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
    return QPixmap.fromImage(qimg)

def draw_grid(img, spacing=40):
    h, w = img.shape[:2]
    for x in range(0, w, spacing):
        cv2.line(img, (x, 0), (x, h), (255, 255, 255), 1)
    for y in range(0, h, spacing):
        cv2.line(img, (0, y), (w, y), (255, 255, 255), 1)
    return img

def euclidean(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def draw_black_grid(img, spacing_px=40):
    """Draws a black grid spaced by spacing_px pixels."""
    h, w = img.shape[:2]
    for x in range(0, w, spacing_px):
        cv2.line(img, (x, 0), (x, h), (255, 255, 255), 1)
    for y in range(0, h, spacing_px):
        cv2.line(img, (0, y), (w, y), (255, 255, 255), 1)
    return img


def classify_perspective(p1, p2):
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    if dx < 30:  # shoulders too close, probably aside
        return "Side"

    ratio = dy / dx
    if ratio > 0.4:
        return "Diagonal"
    else:
        return "Front"


# === Dashboard ===
class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Width Dashboard")
        self.resize(1200, 800)

        # Log
        self.log_file = open(os.path.join(script_dir, "../session_log/face_width.txt"), "a", encoding="utf-8")
        os.makedirs(os.path.dirname(self.log_file.name), exist_ok=True)

        # Fullscreen toggle
        toggle_fullscreen = QAction("Toggle Fullscreen", self)
        toggle_fullscreen.setShortcut("F11")
        toggle_fullscreen.triggered.connect(self.toggle_fullscreen)
        self.addAction(toggle_fullscreen)

        # Layout
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        top = QHBoxLayout()

        # Left: image + dropdown
        left = QVBoxLayout()
        self.proc_label = QLabel()
        self.proc_label.setAlignment(Qt.AlignCenter)
        self.proc_label.setFixedSize(512, 512)
        self.proc_label.mousePressEvent = self.on_click

        self.view_selector = QComboBox()
        self.view_selector.addItems(["Unlabeled", "Front", "Diagonal", "Side"])
        self.view_selector.setFixedWidth(150)
        self.view_selector.hide()

        left.addWidget(self.proc_label)
        left.addWidget(self.view_selector, alignment=Qt.AlignCenter)

        # Right: result
        self.result_label = QLabel("Result")
        self.result_label.setAlignment(Qt.AlignCenter)

        top.addLayout(left, 1)
        top.addWidget(self.result_label, 1)

        # Bottom: log
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        layout.addLayout(top)
        layout.addWidget(self.log)

        # State
        self.clicks = []
        self.files = [os.path.join(face_width_input, f) for f in sorted(os.listdir(face_width_input)) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        self.index = 0
        self.last_saved_path = None

        # Resume
        if os.path.exists(progress_file):
            last = open(progress_file).read().strip()
            base_names = [os.path.basename(p) for p in self.files]
            if last in base_names:
                self.index = min(base_names.index(last) + 1, len(self.files) - 1)

        if self.files:
            self.load_image(self.files[self.index])
        else:
            self.add_log("⚠️ No images found in input folder.")

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def load_image(self, filename):
        self.filename = filename
        img = cv2.imread(filename)
        self.raw_img = img.copy()

        if self.raw_img is None:
            self.add_log(f"⚠️ Could not load {filename}")
            return

        self.proc_label.setPixmap(
            cvimg_to_qpix(self.raw_img).scaled(
                self.proc_label.width(), self.proc_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

        self.add_log(
            f"🖼️ Processing: {os.path.basename(filename)}, {self.index+1}-th of {len(self.files)} files "
        )

        preview = draw_grid(img.copy())
        self.proc_label.setPixmap(cvimg_to_qpix(preview).scaled(self.proc_label.width(), self.proc_label.height(), Qt.KeepAspectRatio))

        self.view_selector.setCurrentIndex(0)
        self.view_selector.hide()
        self.clicks = []


    def on_click(self, event):
        pos = event.position().toPoint()

        # store plain coordinates
        self.clicks.append((pos.x(), pos.y()))
        self.add_log(f"Point clicked: {pos.x()}, {pos.y()}")

        step = len(self.clicks)

        if step == 1:
            self.add_log("✅ Leftist lateral neck contour recorded.")
            self.add_log("2️⃣: Click rightist lateral neck contour!")

        if self.raw_img is None:
            return

        # copy image and draw grid
        img_copy = draw_black_grid(self.raw_img.copy(), spacing_px=40)

        # scale factors
        h, w = self.raw_img.shape[:2]
        disp_w, disp_h = self.proc_label.width(), self.proc_label.height()
        scale_x, scale_y = w / disp_w, h / disp_h

        # draw all clicked points so far
        coords = []
        for i, (x, y) in enumerate(self.clicks):
            cx, cy = int(x * scale_x), int(y * scale_y)
            coords.append((cx,cy))

            if i == 0:
                cv2.circle(img_copy, (cx, cy), 5, (0, 255, 0), -1)
            if i == 1:
                cv2.circle(img_copy, (cx, cy), 5, (0, 255, 0), -1)
            if i == 2:
                cv2.circle(img_copy, (cx, cy), 5, (255, 0, 0), -1)
            if i == 3:
                cv2.circle(img_copy, (cx, cy), 5, (255, 0, 0), -1)

            cv2.putText(img_copy, f"{i+1}", (cx, cy+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
            cv2.putText(img_copy, f"{i+1}", (cx, cy+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90,255,255), 2)

        if step == 1:
            cv2.line(img_copy,(cx-400, cy), (cx+400, cy), (150,150,150), 1)

        if step >= 2:
            self.add_log("draw line from p1 to p2")
            p1,p2 = coords[0], coords[1]
            cv2.line(img_copy, p1, p2, (0,255,0), 2)
            cv2.putText(img_copy, "Face width", (p1[0], p1[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
            cv2.putText(img_copy, "Face width", (p1[0], p1[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        if step == 3:
            cv2.line(img_copy,(cx, cy-400), (cx, cy+400), (150,150,150), 1)

        if step >= 4:
            self.add_log("draw line from p3 to p4")
            p3,p4 = coords[2], coords[3]
            cv2.line(img_copy, p3, p4, (255,0,0), 2)
            cv2.putText(img_copy, "Face height", (p3[0], p4[1]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
            cv2.putText(img_copy, "Face height", (p3[0], p4[1]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # update preview
        self.proc_label.setPixmap(
            cvimg_to_qpix(img_copy).scaled(
                self.proc_label.width(), self.proc_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
         )

        if len(self.clicks) == 4:
            self.process_clicks()

    def process_clicks(self):
        img = self.raw_img.copy()
        h, w = img.shape[:2]
        disp_w, disp_h = self.proc_label.width(), self.proc_label.height()
        scale_x, scale_y = w / disp_w, h / disp_h

        def to_coords(p):  # p is a tuple (x, y)
            return int(p[0] * scale_x), int(p[1] * scale_y)

        for i, (x, y) in enumerate(self.clicks):
            cx, cy = int(x * scale_x), int(y * scale_y)
            cv2.putText(img, f"{i+1}", (cx, cy+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
            cv2.putText(img, f"{i+1}", (cx, cy+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90,255,255), 2)

        p1, p2, p3, p4 = [to_coords(p) for p in self.clicks]

        # First line
        cv2.line(img, p1, p2, (0, 255, 0), 2)

        # Draw small circles at first-line start
        cv2.circle(img, p1, 5, (0, 255, 0), -1)

        # Draw small circles at first-line end
        cv2.circle(img, p2, 5, (0, 255, 0), -1)

        # labelling first line
        cv2.putText(img, "Face width", (p1[0], p1[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
        cv2.putText(img, "Face width", (p1[0], p1[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90,255,255), 2)

        # Second line
        cv2.line(img, p3, p4, (255, 0, 0), 2)

        # Draw small circles at second-line start
        cv2.circle(img, p3, 5, (255, 0, 0), -1)

        # Draw small circles at second-line end
        cv2.circle(img, p4, 5, (255, 0, 0), -1)



        # cv2.line(img, p1, p2, (0, 255, 0), 2)
        # cv2.line(img, p3, p4, (255, 0, 0), 2)

        L1 = euclidean(p1, p2)
        L2 = euclidean(p3, p4)


        # labelling second line
        cv2.putText(img, "Face height", (p3[0], int( p3[1] + L2 * 2/3 ) ),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
        cv2.putText(img, "Face height", (p3[0], int( p3[1] + L2 * 2/3 ) ),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

        ratio = L1/ L2 if L1 else 0

        # --- Perspective classification ---
        perspective = classify_perspective(p1, p2)

        # Update combo box automatically
        index_map = {"Unlabeled": 0, "Front": 1, "Diagonal": 2, "Side": 3}
        self.view_selector.setCurrentIndex(index_map[perspective])
        self.add_log(f"🧭 Auto-perspective : {perspective}, 👁️ Please compare with your own visual judgment!")


        cv2.putText(img, f"Ratio: {ratio:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        self.add_log(f"📏 L1: {L1:.2f}, L2: {L2:.2f}, Ratio: {ratio:.2f}")

        if ratio > .75:
            folder = output_folders["high"]
            classification = "Width ( > 0.75 )"
        elif .68 <= ratio <= .75:
            folder = output_folders["mid"]
            classification = "Mid (0.68 - 0.75)"
        elif 0 <= ratio < .68:
            folder = output_folders["low"]
            classification = "Narrow (0 - 0.67)"
        else:
            self.add_log("⚠️ Ratio out of range. Skipped.")
            return

        # Timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        legend_x, legend_y = 0, 0
        cv2.rectangle(img, (legend_x-10, legend_y-20),
                      (legend_x+240, legend_y+110), (0,0,0), -1)
        cv2.rectangle(img, (legend_x-10, legend_y-20),
                      (legend_x+240, legend_y+110), (255,255,255), 1)

        cv2.putText(img, "Legend:", (legend_x, legend_y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.putText(img, "Green = Face width", (legend_x, legend_y+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.putText(img, "Blue = Face height", (legend_x, legend_y+35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        # Classification + ratio + timestamp
        cv2.putText(img, f"Result: {classification}", (legend_x, legend_y+55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.putText(img, f"Ratio: {ratio:.2f}", (legend_x, legend_y+75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.putText(img, f"Time: {timestamp}", (legend_x, legend_y+95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 2)


        # save result
        out_path = os.path.join(folder, os.path.basename(self.filename))
        cv2.imwrite(out_path, img)
        self.last_saved_path = out_path
        self.result_label.setPixmap(cvimg_to_qpix(img).scaled(self.result_label.width(), self.result_label.height(), Qt.KeepAspectRatio))
        self.add_log(f"✅ Saved to {out_path}")
        open(progress_file, "w").write(os.path.basename(self.filename))
        self.view_selector.show()

    def add_log(self, text):
        cursor = self.log.textCursor()
        cursor.movePosition(QTextCursor.Start)
        cursor.insertText(text + "\n")
        self.log_file.write(text + "\n")
        self.log_file.flush()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            # If a wrong result was already saved, remove it
            if hasattr(self, "last_saved_path") and self.last_saved_path and os.path.exists(self.last_saved_path):
                try:
                    os.remove(self.last_saved_path)
                    self.add_log(f"🗑️ Removed wrong result: {self.last_saved_path}")
                except Exception as e:
                    self.add_log(f"⚠️ Could not remove wrong result: {e}")
                self.last_saved_path = None

            # Reset state and reload current image
            if self.filename:
                self.load_image(self.filename)
                self.add_log("↩️ Reset current image. Please click again.")

            self.clicks = []
            self.processed = False
            self.last_tilt = None
        elif event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter, Qt.Key_Right):
            self.index = min(self.index + 1, len(self.files) - 1)
            self.load_image(self.files[self.index])
        elif event.key() in (Qt.Key_Backspace, Qt.Key_Left):
            self.index = max(self.index - 1, 0)
            self.load_image(self.files[self.index])
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.log_file.close()
        super().closeEvent(event)


# === Main ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    dash = Dashboard()
    dash.show()
    sys.exit(app.exec())


