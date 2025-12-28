

!apt-get install -y tesseract-ocr
!pip install -q transformers==4.37.2 accelerate sentencepiece
!pip install pytesseract pillow opencv-python matplotlib pyngrok==7.0.0 flask requests sentence-transformers==2.2.0 supabase -q

import os, re, difflib, tempfile, subprocess, cv2, numpy as np, torch
from PIL import Image
from flask import Flask, request, jsonify
from pyngrok import ngrok
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from supabase import create_client
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# ======================================
# MOUNT DRIVE & SUPABASE INIT
# ======================================

SUPABASE_URL = "YOUR_SUPABASE_URL"
SUPABASE_KEY = "YOUR_SUPABASE_KEY"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)

# ======================================
# LOAD FINETUNED MODEL
# ======================================
FINETUNED_MODEL_DIR = "/content/drive/YOUR_FINETUNED_MODEL_OUTPUT_DIRECTORY_PATH"
try:
    processor = TrOCRProcessor.from_pretrained(FINETUNED_MODEL_DIR)
    model = VisionEncoderDecoderModel.from_pretrained(FINETUNED_MODEL_DIR)
    print("✅ Fine-tuned TrOCR loaded.")
except Exception as e:
    print("⚠️ Using base model due to error:", e)
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("🚀 Device:", device)

# =========================================
# 🧩 Stable Line Extraction (Noise-resistant)
# =========================================
def extract_lines_with_padding(pil_image, pad=15):
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img.shape[1] // 25, 7))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    sizes = stats[1:, -1]
    img_clean = np.zeros(output.shape, dtype=np.uint8)
    min_size = 0.001 * img.shape[0] * img.shape[1]
    for i in range(nb_components - 1):
        if sizes[i] >= min_size:
            img_clean[output == i + 1] = 255
    contours, _ = cv2.findContours(img_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    lines, prev_y, prev_h = [], -1, -1
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h < 12 or w < 40:
            continue
        if prev_y != -1 and abs(y - prev_y) < prev_h * 0.6:
            continue
        y1, y2 = max(0, y - 10), min(img.shape[0], y + h + 10)
        x1, x2 = max(0, x - 15), min(img.shape[1], x + w + 15)
        cropped = img[y1:y2, x1:x2]
        padded = cv2.copyMakeBorder(cropped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        lines.append(Image.fromarray(padded))
        prev_y, prev_h = y, h

    print(f"🧾 Detected {len(lines)} clean line(s).")
    return lines

# =========================================
# 🔹 OCR FUNCTIONS
# =========================================
def extract_text_with_trocr(pil_image):
    lines = extract_lines_with_padding(pil_image)
    final_lines = []
    for i, line_img in enumerate(lines):
        pixel_values = processor(images=line_img, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            ids = model.generate(pixel_values, max_length=256, num_beams=3, early_stopping=True)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if text:
            print(f"🧠 Line {i+1}: {text}")
            final_lines.append(text)
    return "\n".join(final_lines)

def extract_text_with_tesseract(pil_image):
    import pytesseract
    return pytesseract.image_to_string(pil_image, config='--oem 3 --psm 6')

# =========================================
# 🔹 Cleaning + Mapping
# =========================================
def clean_text(text):
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join([l.strip() for l in text.splitlines() if l.strip()])
    return text.strip()

def get_mapping_from_supabase():
    data = supabase.table("ocr_data").select("*").execute()
    mapping = {}
    for row in data.data:
        mapping[row["label"]] = row.get("ocr_values", [])
    return mapping

def normalize_code_text(ocr_text, mapping):
    token_pattern = r"[A-Za-z_]+|\d+|[^\w\s]"
    normalized_lines = []
    for line in ocr_text.splitlines():
        tokens = re.findall(token_pattern, line)
        cleaned = []
        for token in tokens:
            lower = token.lower()
            match = None
            for label, variants in mapping.items():
                if lower in [v.lower() for v in variants]:
                    match = label
                    break
            if not match:
                best_score, best_label = 0, None
                for label, variants in mapping.items():
                    for v in variants:
                        score = difflib.SequenceMatcher(None, lower, v.lower()).ratio()
                        if score > best_score:
                            best_score, best_label = score, label
                if best_label and best_score > 0.75:
                    match = best_label
            cleaned.append(match if match else token)
        normalized_lines.append(" ".join(cleaned))
    return "\n".join(normalized_lines)

# =========================================
# 🔹 Code Execution Utility
# =========================================
def run_cmd(cmd, input_data=None, cwd=None, timeout=5):
    try:
        proc = subprocess.run(cmd, input=input_data.encode() if isinstance(input_data, str) else input_data,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, timeout=timeout)
        return proc.returncode, proc.stdout.decode(), proc.stderr.decode()
    except subprocess.TimeoutExpired:
        return 124, "", "Execution timed out"
    except Exception as e:
        return 1, "", str(e)

# =========================================
# 🔹 ROUTES
# =========================================
@app.route("/process", methods=["POST"])
def process_image():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400
    pil_image = Image.open(file).convert("RGB")
    text = extract_text_with_trocr(pil_image)
    used_model = "🧠 Fine-tuned TrOCR"
    if not text.strip() or len(text.strip()) < 5:
        text = extract_text_with_tesseract(pil_image)
        used_model = "🔍 Tesseract OCR"
    raw_text = text
    cleaned_text = clean_text(raw_text)
    mapping = get_mapping_from_supabase()
    mapped_text = normalize_code_text(cleaned_text, mapping)
    return jsonify({
        "usedModel": used_model,
        "rawText": raw_text,
        "cleanedText": cleaned_text,
        "mappedText": mapped_text
    })

@app.route("/run", methods=["POST"])
def run_code():
    import shutil
    data = request.get_json(force=True)
    language = data.get("language")
    code = data.get("code", "")
    stdin = data.get("stdin", "")

    if language not in ("python", "java", "c"):
        return jsonify({"error": "Unsupported language"}), 400

    # Colab-friendly: ensure javac/java exist
    if language == "java":
        if not shutil.which("javac") or not shutil.which("java"):
            # Install default JDK if missing
            rc_install, out_install, err_install = run_cmd(
                ["apt-get", "update"]
            )
            rc_jdk, out_jdk, err_jdk = run_cmd(
                ["apt-get", "install", "-y", "default-jdk"]
            )
            if rc_jdk != 0:
                return jsonify({
                    "exit_code": rc_jdk,
                    "stdout": out_jdk,
                    "stderr": err_jdk or "Failed to install Java JDK"
                })

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            if language == "python":
                src = os.path.join(tmpdir, "main.py")
                open(src, "w").write(code)
                rc, out, err = run_cmd(["python3", src], input_data=stdin, cwd=tmpdir)

            elif language == "c":
                src = os.path.join(tmpdir, "main.c")
                bin_path = os.path.join(tmpdir, "a.out")
                open(src, "w").write(code)
                rc_c, out_c, err_c = run_cmd(["gcc", src, "-O2", "-std=c11", "-o", bin_path])
                if rc_c != 0:
                    return jsonify({"exit_code": rc_c, "stdout": out_c, "stderr": err_c})
                rc, out, err = run_cmd([bin_path], input_data=stdin, cwd=tmpdir)

            else:  # Java
                src = os.path.join(tmpdir, "Main.java")
                open(src, "w").write(code)

                # Compile
                rc_j, out_j, err_j = run_cmd(["javac", "Main.java"], cwd=tmpdir)
                if rc_j != 0:
                    return jsonify({"exit_code": rc_j, "stdout": out_j, "stderr": err_j})

                # Run with stdin
                rc, out, err = run_cmd(["java", "Main"], input_data=stdin, cwd=tmpdir)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"exit_code": rc, "stdout": out, "stderr": err})


# =========================================
# 🔹 Update Mapping + Okapi Routes
# =========================================
TOKEN_REGEX = r"[A-Za-z_]\w*|\d+|[^\s]"

def align_line_chunks(raw_line, corrected_line, line_no):
    raw_tokens = raw_line.split()
    corr_tokens = corrected_line.split()
    mappings = []
    for i, (r, c) in enumerate(zip(raw_tokens, corr_tokens)):
        mappings.append({
            "label": c,
            "variant": r,
            "type": "equal" if r == c else "change",
            "line": line_no
        })
    return mappings

@app.route("/update", methods=["POST"])
def update_mapping():
    data = request.get_json(force=True)
    raw_text = (data.get("rawText", "") or "").strip()
    corrected_code = (data.get("correctedCode", "") or "").strip()
    mapped_code = (data.get("mappedCode", "") or "").strip()
    if not raw_text or not corrected_code or not mapped_code:
        return jsonify({"error": "Missing fields"}), 400
    raw_lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    corrected_lines = [l.strip() for l in corrected_code.splitlines() if l.strip()]
    mapped_lines = [l.strip() for l in mapped_code.splitlines() if l.strip()]
    all_mappings = []
    for idx, (r, m, c) in enumerate(zip(raw_lines, mapped_lines, corrected_lines), start=1):
        if m != c:
            all_mappings.extend(align_line_chunks(r, c, line_no=idx))
    changed = [m for m in all_mappings if m["type"] == "change"]
    return jsonify({"message": "Processed", "changedMappings": changed})

@app.route("/okapi", methods=["POST"])
def okapi():
    data = request.get_json(force=True)
    changes = data.get("changes", [])
    if not changes:
        return jsonify({"error": "No changes"}), 400
    all_labels_response = supabase.table("ocr_data").select("*").execute()
    all_labels = {r["label"]: r["ocr_values"] or [] for r in all_labels_response.data}
    updated_records = []
    for change in changes:
        label, variant = change["label"], change["variant"]
        conflict = any(variant in v and lbl != label for lbl, v in all_labels.items())
        if conflict: continue
        existing = supabase.table("ocr_data").select("*").eq("label", label).execute()
        if existing.data:
            row = existing.data[0]
            vals = row.get("ocr_values", []) or []
            if variant not in vals:
                supabase.table("ocr_data").update({"ocr_values": vals + [variant]}).eq("id", row["id"]).execute()
                updated_records.append({"label": label, "added_variant": variant})
        else:
            supabase.table("ocr_data").insert({"label": label, "ocr_values": [variant]}).execute()
            updated_records.append({"label": label, "created_with_variant": variant})
    return jsonify({"message": "DB updated", "updates": updated_records})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# =========================================
# 🔹 START SERVER
# =========================================
ngrok.set_auth_token("YOUR_NGROK_TOKEN")
public_url = ngrok.connect(5000, "http")
print(" * Ngrok tunnel URL:", public_url)
app.run(port=5000, use_reloader=False)
