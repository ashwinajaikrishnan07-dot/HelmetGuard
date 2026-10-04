"""
HelmetGuard AI - AI-Powered Traffic Helmet Violation Detection

Run with:
    python -m streamlit run app.py

Pipeline:
    image -> YOLOv8s (best.pt, imgsz=960, conf=0.10, iou=0.45, augment=True)
          -> raw predictions
          -> class-specific filtering (no_helmet >= 0.45, helmet >= 0.30)
          -> filtered detections (single source of truth)
          -> boxes / counts / metrics / CSV / annotated image
"""

import io
import textwrap
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError


# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "best.pt"

# Model was trained at 640; 960 at inference helps with small riders.
IMG_SIZE = 960
YOLO_CONF_DEFAULT = 0.10   # internal YOLO inference threshold
YOLO_IOU = 0.45
YOLO_AUGMENT = True

VIOLATION_CLASS = "no_helmet"
HELMET_CLASS = "helmet"
EXPECTED_CLASSES = {0: HELMET_CLASS, 1: VIOLATION_CLASS}

# Application-level display thresholds (applied AFTER YOLO inference)
NO_HELMET_MIN_CONF = 0.45
HELMET_MIN_CONF = 0.30
CLASS_MIN_CONF = {
    VIOLATION_CLASS: NO_HELMET_MIN_CONF,
    HELMET_CLASS: HELMET_MIN_CONF,
}

RED_BGR = (40, 40, 220)
GREEN_BGR = (60, 170, 40)


# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="HelmetGuard AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ================================================================
# HTML RENDERING
# ================================================================

def render_html(html: str, container=st):
    """
    Render custom HTML safely.

    Markdown treats lines indented by 4+ spaces (after a blank line) as a
    code block, which made raw tags appear on screen. Dedent, strip every
    line and drop blank lines so the HTML is always rendered as HTML.
    """
    lines = textwrap.dedent(html).strip().splitlines()
    flat = "\n".join(line.strip() for line in lines if line.strip())
    container.markdown(flat, unsafe_allow_html=True)


# ================================================================
# CUSTOM CSS
# ================================================================

render_html(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Inter', sans-serif; }
    .stApp { background: #f5f7fb; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1250px; }
    #MainMenu, footer { visibility: hidden; }

    /* HEADER */
    .hg-header { background: linear-gradient(135deg, #0b1f44 0%, #163a7a 100%); color: #fff;
        border-radius: 18px; padding: 28px 34px; margin-bottom: 26px;
        box-shadow: 0 8px 24px rgba(11,31,68,.18); }
    .hg-title { font-size: 2.3rem; font-weight: 800; margin: 0; letter-spacing: -.5px; color: #fff; }
    .hg-sub { font-size: 1.1rem; font-weight: 500; color: #c9d6f2; margin: 4px 0 12px; }
    .hg-desc { font-size: .97rem; color: #e3eaf8; margin: 0 0 14px; }
    .hg-badge { display: inline-block; background: rgba(255,255,255,.14);
        border: 1px solid rgba(255,255,255,.3); color: #fff; padding: 5px 13px;
        border-radius: 999px; font-size: .78rem; font-weight: 600; letter-spacing: .4px; }

    /* SECTION */
    .hg-section { font-size: 1.15rem; font-weight: 700; color: #0b1f44; margin: 22px 0 10px; }

    /* CARDS */
    .hg-card { background: #fff; border-radius: 14px; padding: 20px 22px;
        border: 1px solid #e4e9f2; box-shadow: 0 2px 10px rgba(16,32,72,.05); }
    .hg-empty { text-align: center; padding: 56px 20px; }
    .hg-empty .icon { font-size: 3rem; }
    .hg-empty h3 { color: #0b1f44; margin: 8px 0 6px; font-weight: 700; }
    .hg-empty p { color: #5b6b88; margin: 0; }

    /* RESULT */
    .hg-result { border-radius: 14px; padding: 22px 26px; margin: 8px 0 18px; }
    .hg-result h2 { margin: 0 0 4px; font-size: 1.5rem; font-weight: 800; padding: 0; }
    .hg-result p { margin: 0; font-size: 1.02rem; }
    .hg-danger { background: #fdecec; border: 1px solid #f5b5b5; border-left: 6px solid #d62828; }
    .hg-danger h2 { color: #b71c1c; } .hg-danger p { color: #6e1a1a; }
    .hg-neutral { background: #eef3fb; border: 1px solid #c4d3ee; border-left: 6px solid #163a7a; }
    .hg-neutral h2 { color: #0b1f44; } .hg-neutral p { color: #33415c; }
    .hg-warning { background: #fff8e6; border: 1px solid #f0d58a; border-left: 6px solid #e0a400; }
    .hg-warning h2 { color: #9a6a00; } .hg-warning p { color: #765400; }

    /* METRICS */
    .hg-metric { background: #fff; border: 1px solid #e4e9f2; border-radius: 14px;
        padding: 18px 20px; box-shadow: 0 2px 10px rgba(16,32,72,.05); height: 100%; }
    .hg-metric .label { font-size: .85rem; color: #5b6b88; font-weight: 600; }
    .hg-metric .value { font-size: 2rem; color: #0b1f44; font-weight: 800; margin-top: 4px; }
    .hg-metric .note { font-size: .74rem; color: #8a97ad; margin-top: 2px; }
    .hg-metric.red { border-top: 4px solid #d62828; } .hg-metric.red .value { color: #c62828; }
    .hg-metric.blue { border-top: 4px solid #163a7a; }
    .hg-metric.gray { border-top: 4px solid #9aa6bb; }

    /* VIOLATION CARD */
    .hg-vcard { background: #fff; border: 1px solid #f1c9c9; border-left: 5px solid #d62828;
        border-radius: 12px; padding: 14px 16px; margin-bottom: 12px; }
    .hg-vcard .t { font-weight: 700; color: #b71c1c; margin-bottom: 4px; }
    .hg-vcard .r { font-size: .9rem; color: #33415c; }
    .hg-imgcap { font-weight: 700; color: #0b1f44; margin-bottom: 6px; }

    /* FOOTER */
    .hg-footer { text-align: center; color: #8a97ad; font-size: .85rem; margin-top: 40px;
        padding-top: 18px; border-top: 1px solid #e4e9f2; }
    .hg-footer b { color: #5b6b88; }

    /* SIDEBAR - readable dark text on white */
    section[data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #e4e9f2; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #0b1f44 !important; }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] { color: #33415c !important; }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #5b6b88 !important; }
    section[data-testid="stSidebar"] [data-testid="stSliderThumbValue"] { color: #163a7a !important; font-weight: 700; }
    section[data-testid="stSidebar"] hr { border-color: #e4e9f2; }
    .hg-side-info { background: #f5f7fb; border: 1px solid #e4e9f2; border-radius: 10px;
        padding: 12px 14px; font-size: .88rem; color: #33415c; }
    .hg-side-info div { margin: 3px 0; color: #33415c; }
    .hg-side-info b { color: #0b1f44; }
    .hg-side-info code { background: #e4e9f2; color: #0b1f44; padding: 1px 5px; border-radius: 4px; font-size: .8rem; }

    /* BUTTONS */
    .stButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 600; padding: .55rem 1.1rem; }
    .stButton > button[kind="primary"] { background: #163a7a; border-color: #163a7a; color: #fff; }
    .stButton > button[kind="primary"]:hover { background: #0b1f44; border-color: #0b1f44; }
    </style>
    """
)


# ================================================================
# MODEL
# ================================================================

@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model(model_path: str):
    """Load the trained YOLO model once and cache it."""
    from ultralytics import YOLO

    return YOLO(model_path)


def verify_model_classes(model) -> bool:
    """The app expects exactly {0: 'helmet', 1: 'no_helmet'}."""
    return {int(k): v for k, v in model.names.items()} == EXPECTED_CLASSES


# ================================================================
# IMAGE LOADING
# ================================================================

def load_image(uploaded_file) -> Image.Image:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


# ================================================================
# YOLO INFERENCE
# ================================================================

def process_image(model, image: Image.Image, conf: float):
    """Run YOLO inference. Boxes come back in original-image pixel coordinates."""
    results = model.predict(
        source=image,
        imgsz=IMG_SIZE,
        conf=conf,
        iou=YOLO_IOU,
        augment=YOLO_AUGMENT,
        verbose=False,
    )
    return results[0]


# ================================================================
# DETECTION EXTRACTION + CLASS-SPECIFIC FILTERING
# ================================================================

def extract_raw_detections(result):
    """All predictions returned by YOLO, BEFORE app-level filtering (debug only)."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    names = result.names
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    raw = []
    for box, confidence, class_id in zip(xyxy, confs, clss):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        raw.append({
            "class": names.get(int(class_id), str(class_id)),
            "confidence": float(confidence),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })

    raw.sort(key=lambda d: d["confidence"], reverse=True)
    return raw


def passes_threshold(det) -> bool:
    """Class-specific display threshold. Unknown classes never pass."""
    min_conf = CLASS_MIN_CONF.get(det["class"])
    return min_conf is not None and det["confidence"] >= min_conf


def filter_detections(raw_detections):
    """
    SINGLE SOURCE OF TRUTH for displayed detections.

    no_helmet >= NO_HELMET_MIN_CONF, helmet >= HELMET_MIN_CONF.
    Everything downstream (boxes, counts, metrics, CSV) uses this output.
    """
    return [d for d in raw_detections if passes_threshold(d)]


def extract_violations(filtered):
    """Only filtered no_helmet detections count as violations."""
    return [d for d in filtered if d["class"] == VIOLATION_CLASS]


def extract_helmets(filtered):
    return [d for d in filtered if d["class"] == HELMET_CLASS]


# ================================================================
# DRAWING
# ================================================================

def _draw_box(img, det, color, label, show_conf):
    h, w = img.shape[:2]
    thickness = max(2, int(round(min(h, w) / 250)))
    font_scale = max(0.5, min(h, w) / 1100)
    font_thick = max(1, thickness - 1)
    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    text = f"{label} - {det['confidence'] * 100:.0f}%" if show_conf else label
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
    pad = 6

    # Label above the box; inside the top edge if there's no room
    ty1 = y1 - th - base - 2 * pad
    if ty1 < 0:
        ty1 = y1
    tx2 = min(w - 1, x1 + tw + 2 * pad)
    ty2 = ty1 + th + base + 2 * pad

    cv2.rectangle(img, (x1, ty1), (tx2, ty2), color, -1)
    cv2.putText(img, text, (x1 + pad, ty2 - pad - base), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)


def draw_detections(image, violations, helmets=None, show_conf=True):
    """Draw ONLY filtered detections: red = no helmet, green = helmet."""
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Subtle red tint inside violation regions
    if violations:
        overlay = img.copy()
        for d in violations:
            cv2.rectangle(overlay, (d["x1"], d["y1"]), (d["x2"], d["y2"]), RED_BGR, -1)
        img = cv2.addWeighted(overlay, 0.12, img, 0.88, 0)

    for d in helmets or []:
        _draw_box(img, d, GREEN_BGR, "HELMET", show_conf)
    for d in violations:
        _draw_box(img, d, RED_BGR, "NO HELMET", show_conf)

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# ================================================================
# STATISTICS
# ================================================================

def calculate_statistics(raw_detections, violations):
    """Violation metrics use filtered detections; total objects uses raw YOLO output."""
    count = len(violations)
    avg_conf = float(np.mean([v["confidence"] for v in violations])) * 100 if count else 0.0
    return {
        "violations": count,
        "avg_conf": avg_conf,
        "total_objects": len(raw_detections),
    }


# ================================================================
# CSV REPORT
# ================================================================

def create_report(violations) -> bytes:
    """CSV of filtered no_helmet violations only."""
    columns = ["Violation ID", "Class", "Confidence", "X1", "Y1", "X2", "Y2"]
    rows = [
        {
            "Violation ID": i,
            "Class": v["class"],
            "Confidence": round(v["confidence"], 4),
            "X1": v["x1"], "Y1": v["y1"], "X2": v["x2"], "Y2": v["y2"],
        }
        for i, v in enumerate(violations, start=1)
    ]
    return pd.DataFrame(rows, columns=columns).to_csv(index=False).encode("utf-8")


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# ================================================================
# UI COMPONENTS
# ================================================================

def metric_card(label, value, style="blue", note=""):
    render_html(f"""
        <div class="hg-metric {style}">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="note">{note}</div>
        </div>
    """)


def result_card(style, title, message):
    render_html(f"""
        <div class="hg-result {style}">
            <h2>{title}</h2>
            <p>{message}</p>
        </div>
    """)


def section_title(text):
    render_html(f'<div class="hg-section">{text}</div>')


def render_header():
    render_html("""
        <div class="hg-header">
            <p class="hg-title">🚦 HelmetGuard AI</p>
            <p class="hg-sub">AI-Powered Traffic Helmet Violation Detection</p>
            <p class="hg-desc">Upload a traffic image to automatically detect and localize riders who are not wearing helmets.</p>
            <span class="hg-badge">YOLO DEEP LEARNING MODEL</span>
        </div>
    """)


def render_empty_state():
    render_html("""
        <div class="hg-card hg-empty">
            <div class="icon">📷</div>
            <h3>Upload a traffic image</h3>
            <p>HelmetGuard AI will analyze the image and identify riders without helmets.</p>
        </div>
    """)


def render_footer():
    render_html("""
        <div class="hg-footer">
            <b>HelmetGuard AI</b> • Deep Learning-Based Traffic Safety System<br>
            Powered by YOLO Object Detection
        </div>
    """)


def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Detection Settings")

        conf = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=0.90,
            value=YOLO_CONF_DEFAULT,
            step=0.05,
            help="YOLO inference threshold. Predictions below this are discarded by YOLO itself.",
        )
        st.caption(
            f"This is the YOLO inference threshold. Final display thresholds are fixed: "
            f"No Helmet ≥ {NO_HELMET_MIN_CONF:.2f}, Helmet ≥ {HELMET_MIN_CONF:.2f}."
        )

        show_helmets = st.checkbox(
            "Show helmet detections", value=True,
            help="Draw green boxes around reliable helmet detections.",
        )
        show_conf = st.checkbox("Show confidence scores", value=True)

        st.markdown("---")
        st.markdown("### Model Information")
        render_html("""
            <div class="hg-side-info">
                <div><b>Model:</b> YOLOv8s</div>
                <div><b>Task:</b> Object Detection</div>
                <div><b>Target:</b> Helmet Violation Detection</div>
            </div>
        """)
        st.caption("🟥 Red = no helmet  ·  🟩 Green = helmet")

    return conf, show_helmets, show_conf


def render_debug_sidebar(model, conf):
    classes = {int(k): v for k, v in model.names.items()}
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🛠️ Developer Info")
        render_html(f"""
            <div class="hg-side-info">
                <div><b>Loaded Model:</b> {MODEL_PATH.name}</div>
                <div><b>Classes:</b> <code>{classes}</code></div>
                <div><b>Inference Size:</b> {IMG_SIZE}</div>
                <div><b>YOLO Confidence:</b> {conf:.2f}</div>
                <div><b>IoU / Augment:</b> {YOLO_IOU} / {YOLO_AUGMENT}</div>
                <div><b>No-Helmet Display Threshold:</b> {NO_HELMET_MIN_CONF:.2f}</div>
                <div><b>Helmet Display Threshold:</b> {HELMET_MIN_CONF:.2f}</div>
            </div>
        """)


def render_raw_predictions(raw_detections):
    with st.expander("🔍 Raw YOLO Predictions", expanded=False):
        st.caption("Debug only: predictions BEFORE class-specific filtering. "
                   "These are not the displayed results.")
        if not raw_detections:
            st.write("YOLO returned zero predictions.")
            return
        df = pd.DataFrame([
            {
                "Class": d["class"],
                "Confidence": round(d["confidence"], 3),
                "Min Required": CLASS_MIN_CONF.get(d["class"], float("nan")),
                "Displayed": "✅" if passes_threshold(d) else "—",
                "Box (x1, y1, x2, y2)": f'{d["x1"]}, {d["y1"]}, {d["x2"]}, {d["y2"]}',
            }
            for d in raw_detections
        ])
        st.dataframe(df, hide_index=True, use_container_width=True)


# ================================================================
# MAIN APPLICATION
# ================================================================

def main():
    conf, show_helmets, show_conf = render_sidebar()
    render_header()

    # ---- Model --------------------------------------------------
    if not MODEL_PATH.is_file():
        st.error("Model file not found. Please place best.pt inside the model folder.")
        st.caption(f"Expected location: {MODEL_PATH}")
        render_footer()
        return

    try:
        model = load_model(str(MODEL_PATH))
    except ImportError:
        st.error("Ultralytics is not installed. Run: pip install -r requirements.txt")
        render_footer()
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load the model: {exc}")
        render_footer()
        return

    render_debug_sidebar(model, conf)

    if not verify_model_classes(model):
        st.error(f"Unexpected model classes. Expected {EXPECTED_CLASSES}, got {dict(model.names)}.")
        render_footer()
        return

    # ---- Upload -------------------------------------------------
    section_title("Upload Traffic Image")
    uploaded = st.file_uploader(
        "Upload Traffic Image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if uploaded is None:
        render_empty_state()
        render_footer()
        return

    if uploaded.size == 0:
        st.error("The uploaded file is empty. Please upload a valid image.")
        render_footer()
        return

    try:
        image = load_image(uploaded)
    except (UnidentifiedImageError, OSError):
        st.error("Invalid image file. Please upload a valid JPG, JPEG or PNG image.")
        render_footer()
        return

    # Reset results when the file or YOLO threshold changes
    file_key = (uploaded.name, uploaded.size, conf)
    if st.session_state.get("file_key") != file_key:
        st.session_state.file_key = file_key
        st.session_state.result = None

    # ---- Analyze ------------------------------------------------
    if st.session_state.result is None:
        _, mid, _ = st.columns([1, 1.4, 1])
        with mid:
            st.image(image, caption=f"Preview · {image.width}×{image.height}px",
                     use_container_width=True)
            analyze = st.button("🔍 Analyze Image", type="primary", use_container_width=True)

        if not analyze:
            render_footer()
            return

        with st.spinner("Running YOLO inference..."):
            try:
                st.session_state.result = process_image(model, image, conf)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Model inference failed: {exc}")
                render_footer()
                return
        st.rerun()

    # ---- Pipeline: raw -> filtered -> everything else -----------
    result = st.session_state.result
    raw_detections = extract_raw_detections(result)
    filtered = filter_detections(raw_detections)
    violations = extract_violations(filtered)
    helmets = extract_helmets(filtered) if show_helmets else []
    stats = calculate_statistics(raw_detections, violations)
    annotated = draw_detections(image, violations, helmets, show_conf)

    # ---- Result status ------------------------------------------
    n = stats["violations"]
    if n > 0:
        noun = "violation" if n == 1 else "violations"
        result_card("hg-danger", "🚨 HELMET VIOLATION DETECTED",
                    f"<b>{n}</b> potential helmet {noun} detected.")
    elif stats["total_objects"] > 0:
        result_card("hg-neutral", "🔎 NO NO-HELMET DETECTION FOUND",
                    "The model detected objects, but none were classified as no-helmet "
                    "above the required confidence threshold.")
    else:
        result_card("hg-warning", "⚠️ NO OBJECTS DETECTED",
                    "The model did not detect any objects at the selected confidence threshold.")

    # ---- Metrics ------------------------------------------------
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("🚨 Detected Helmet Violations", n, "red" if n else "blue",
                    f"no-helmet ≥ {NO_HELMET_MIN_CONF:.0%}")
    with m2:
        metric_card("🎯 Avg Confidence", f"{stats['avg_conf']:.1f}%" if n else "—",
                    "blue", "across no-helmet detections")
    with m3:
        metric_card("📍 Locations Found", n, "blue", "localized with bounding boxes")
    with m4:
        metric_card("📦 Total Detected Objects", stats["total_objects"], "gray",
                    "all classes · not a rider count")

    # ---- Image comparison ---------------------------------------
    section_title("Detection Comparison")
    left, right = st.columns(2)
    with left:
        render_html('<div class="hg-imgcap">Original Image</div>')
        st.image(image, use_container_width=True)
    with right:
        render_html('<div class="hg-imgcap">AI Detection Result</div>')
        st.image(annotated, use_container_width=True)

    # ---- Violation details --------------------------------------
    section_title("Violation Details")
    if n:
        cols = st.columns(3)
        for i, v in enumerate(violations, start=1):
            with cols[(i - 1) % 3]:
                render_html(f"""
                    <div class="hg-vcard">
                        <div class="t">Violation #{i}</div>
                        <div class="r"><b>Confidence:</b> {v["confidence"] * 100:.1f}%</div>
                        <div class="r"><b>Location:</b> ({v["x1"]}, {v["y1"]}) → ({v["x2"]}, {v["y2"]})</div>
                    </div>
                """)
    elif stats["total_objects"] == 0:
        st.info("No objects were detected at the selected YOLO confidence threshold.")
    else:
        st.info(f"No no-helmet detection reached the {NO_HELMET_MIN_CONF:.2f} display threshold.")

    render_raw_predictions(raw_detections)

    # ---- Export -------------------------------------------------
    section_title("Export Results")
    stem = Path(uploaded.name).stem
    d1, d2, _ = st.columns([1, 1, 2])
    with d1:
        st.download_button("⬇️ Download Annotated Image", image_to_png_bytes(annotated),
                           file_name=f"{stem}_helmetguard.png", mime="image/png",
                           use_container_width=True)
    with d2:
        st.download_button("📄 Download Violation Report", create_report(violations),
                           file_name=f"{stem}_violations.csv", mime="text/csv",
                           use_container_width=True)

    if st.button("↺ Analyze another image"):
        st.session_state.result = None
        st.session_state.file_key = None
        st.rerun()

    render_footer()


if __name__ == "__main__":
    main()
