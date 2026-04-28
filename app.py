import io
import os
import re
from difflib import SequenceMatcher

import easyocr
import fitz
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import streamlit as st

st.set_page_config(page_title="Smart KYC: Clarity at Every Step.", layout="wide")

STEPS = [
    "Document Upload and Form",
    "Review and Submit",
    "Status Tracker and Help",
]


def init_state():
    if "step" not in st.session_state:
        st.session_state.step = 0
    if "form" not in st.session_state:
        st.session_state.form = {
            "citizenship_number": "",
            "full_name": "",
            "dob": "",
            "district": "",
            "municipality": "",
            "ward_number": "",
        }
    if "status" not in st.session_state:
        st.session_state.status = "Draft"
    if "ocr_data" not in st.session_state:
        st.session_state.ocr_data = None
    if "ocr_applied_for" not in st.session_state:
        st.session_state.ocr_applied_for = None
    if "ocr_file_name" not in st.session_state:
        st.session_state.ocr_file_name = None


MONTH_MAP = {
    "JAN": "01",
    "FEB": "02",
    "MAR": "03",
    "APR": "04",
    "MAY": "05",
    "JUN": "06",
    "JUL": "07",
    "AUG": "08",
    "SEP": "09",
    "SEPT": "09",
    "OCT": "10",
    "NOV": "11",
    "DEC": "12",
}


@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(["en"], gpu=False)


def load_image_from_bytes(file_bytes, file_ext):
    if file_ext == ".pdf":
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=250)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def preprocess_image(image):
    gray = ImageOps.autocontrast(image.convert("L"))
    width, height = gray.size
    max_dim = max(width, height)
    if max_dim < 1200:
        scale = 1200 / max_dim
        gray = gray.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    gray = gray.filter(ImageFilter.UnsharpMask(radius=1.5, percent=180, threshold=3))
    return gray


@st.cache_data(show_spinner=False)
def extract_text_lines(file_bytes, file_ext, enhance_image):
    image = load_image_from_bytes(file_bytes, file_ext)
    if enhance_image:
        image = preprocess_image(image)
    reader = get_ocr_reader()
    results = reader.readtext(np.array(image), detail=1, paragraph=False)
    if not results:
        return [], 0.0, []
    lines = [text for _, text, conf in results if text.strip() and conf >= 0.3]
    avg_conf = sum(conf for _, _, conf in results) / len(results)
    return lines, avg_conf, results


def normalize_month(value):
    if not value:
        return ""
    value = value.strip()
    if value.isdigit():
        return value.zfill(2)
    return MONTH_MAP.get(value.upper()[:4], "")


def parse_date_from_text(text):
    match = re.search(
        r"Year[:\s]*([0-9]{4}).*?Month[:\s]*([A-Za-z]{3,9}|[0-9]{1,2}).*?Day[:\s]*([0-9]{1,2})",
        text,
        re.IGNORECASE,
    )
    if match:
        year, month_raw, day = match.groups()
        month = normalize_month(month_raw)
        if month:
            return f"{year}-{month}-{day.zfill(2)}"

    match = re.search(r"([0-9]{4})[/-]([0-9]{1,2})[/-]([0-9]{1,2})", text)
    if match:
        year, month, day = match.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

    return ""


def find_value_after_label(lines, regexes):
    for index, line in enumerate(lines):
        for regex in regexes:
            match = re.search(regex, line, re.IGNORECASE)
            if not match:
                continue

            value = line[match.end():].strip(" :.-")
            if value:
                return value

            if index + 1 < len(lines):
                next_line = lines[index + 1].strip(" :.-")
                if next_line:
                    return next_line
    return ""


def normalize_label(text):
    return re.sub(r"[^a-z]", "", text.lower())


def label_matches(text, keywords):
    normalized = normalize_label(text)
    for keyword in keywords:
        if keyword in normalized:
            return True
        if SequenceMatcher(None, normalized, keyword).ratio() >= 0.82:
            return True
    return False


def bbox_center(bbox):
    xs = [point[0] for point in bbox]
    ys = [point[1] for point in bbox]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def bbox_height(bbox):
    ys = [point[1] for point in bbox]
    return max(ys) - min(ys)


def find_value_right_of_label(
    ocr_items,
    label_keywords,
    min_conf=0.3,
    value_extractor=None,
    validator=None,
):
    candidates = []
    heights = [item["height"] for item in ocr_items]
    avg_height = sum(heights) / len(heights) if heights else 0
    tolerance = avg_height * 0.8 if avg_height else 15

    for item in ocr_items:
        if item["conf"] < min_conf:
            continue
        text = item["text"]
        if not label_matches(text, label_keywords):
            continue

        if ":" in text:
            value = text.split(":", 1)[1].strip(" :.-")
            if value_extractor:
                value = value_extractor(value)
            if value and (validator is None or validator(value)):
                return value

        label_x, label_y = item["center"]
        for candidate in ocr_items:
            if candidate["conf"] < min_conf:
                continue
            if candidate["center"][0] <= label_x:
                continue
            if abs(candidate["center"][1] - label_y) > tolerance:
                continue
            if label_matches(candidate["text"], label_keywords):
                continue
            candidate_text = candidate["text"]
            if value_extractor:
                candidate_text = value_extractor(candidate_text)
            if not candidate_text:
                continue
            if validator is not None and not validator(candidate_text):
                continue
            candidates.append((candidate["center"][0] - label_x, candidate_text))

    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]
    return ""


def extract_citizenship_number(value):
    if not value:
        return ""
    compact = value.replace(" ", "")
    match = re.search(r"\b[0-9]{1,3}(?:-[0-9]{1,3}){2,4}\b", compact)
    if match:
        return match.group(0)
    return ""


def clean_full_name(value):
    if not value:
        return ""
    value = re.split(r"\bSex\b", value, maxsplit=1, flags=re.IGNORECASE)[0]
    value = re.sub(r"[^A-Za-z\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value.title()


def is_probable_name(value):
    if not value:
        return False
    words = value.split()
    if len(words) < 2:
        return False
    if any(word.lower() in {"sex", "female", "male"} for word in words):
        return False
    return True


def find_all_values_after_label(lines, regex):
    values = []
    for line in lines:
        match = re.search(regex, line, re.IGNORECASE)
        if match:
            value = line[match.end():].strip(" :.-")
            if value:
                values.append(value)
    return values


def parse_citizenship_fields(text_lines, ocr_results=None):
    raw_lines = []
    if ocr_results:
        raw_lines = [text for _, text, _ in ocr_results if text.strip()]
    elif text_lines:
        raw_lines = text_lines
    if not raw_lines:
        return {}

    clean_lines = [re.sub(r"\s+", " ", line).strip() for line in raw_lines if line.strip()]
    joined_text = " ".join(clean_lines)

    ocr_items = []
    if ocr_results:
        for bbox, text, conf in ocr_results:
            if not text.strip():
                continue
            ocr_items.append(
                {
                    "text": text.strip(),
                    "conf": conf,
                    "center": bbox_center(bbox),
                    "height": bbox_height(bbox),
                }
            )

    data = {}
    citizenship_number = ""
    if ocr_items:
        citizenship_number = find_value_right_of_label(
            ocr_items,
            [
                "citizenshipcertificateno",
                "citizenshipcertificate",
                "citizenshipno",
                "citizencertificateno",
                "citizencertificate",
            ],
            value_extractor=extract_citizenship_number,
            validator=lambda value: bool(value),
        )
    if not citizenship_number:
        citizenship_number = extract_citizenship_number(
            find_value_after_label(
                clean_lines,
                [
                    r"Citizenship\s*(Certificate)?\s*No\.?\s*[:]*",
                    r"Citizen\s*Number\s*[:]*",
                ],
            )
        )
    if not citizenship_number:
        citizenship_number = extract_citizenship_number(joined_text)
    if citizenship_number:
        data["citizenship_number"] = citizenship_number

    full_name = ""
    if ocr_items:
        full_name = find_value_right_of_label(
            ocr_items,
            ["fullname", "fullnam"],
            min_conf=0.2,
            value_extractor=clean_full_name,
            validator=is_probable_name,
        )
    if not full_name:
        full_name = clean_full_name(find_value_after_label(clean_lines, [r"Full\s*Name\s*[:]*"]))
    if full_name and is_probable_name(full_name):
        data["full_name"] = full_name
    else:
        stop_words = {
            "government",
            "nepal",
            "citizenship",
            "certificate",
            "birth",
            "place",
            "district",
            "municipality",
            "ward",
            "sex",
        }
        for line in clean_lines:
            if ":" in line:
                continue
            if len(line.split()) < 2:
                continue
            lowered = line.lower()
            if any(word in lowered for word in stop_words):
                continue
            candidate = clean_full_name(line)
            if is_probable_name(candidate):
                data["full_name"] = candidate
                break

    dob = parse_date_from_text(joined_text)
    if dob:
        data["dob"] = dob

    districts = []
    if ocr_items:
        district_value = find_value_right_of_label(ocr_items, ["district"], value_extractor=str.strip)
        if district_value:
            districts = [district_value]
    if not districts:
        districts = find_all_values_after_label(clean_lines, r"District\s*[:]*")
    if districts:
        data["district"] = districts[-1]

    municipality = ""
    if ocr_items:
        municipality = find_value_right_of_label(ocr_items, ["municipality", "vdc"], value_extractor=str.strip)
    if not municipality:
        municipality = find_value_after_label(clean_lines, [r"Municipality\s*[:]*", r"VDC\s*[:]*"])
    if municipality:
        data["municipality"] = municipality

    ward = ""
    if ocr_items:
        ward = find_value_right_of_label(ocr_items, ["wardno", "wardnumber", "ward"], value_extractor=str.strip)
    if not ward:
        ward = find_value_after_label(clean_lines, [r"Ward\s*No\.?\s*[:]*"])
    if ward:
        ward_digits = re.sub(r"[^0-9]", "", ward)
        data["ward_number"] = ward_digits or ward

    return data


def apply_ocr_to_form(form, ocr_data, overwrite=False):
    for key, value in ocr_data.items():
        if key in form and (overwrite or not form[key].strip()):
            form[key] = value


def quality_issues(uploaded_file):
    issues = []
    if uploaded_file is None:
        return issues
    size_kb = uploaded_file.size / 1024
    if size_kb < 80:
        issues.append("Image looks low resolution or heavily compressed")
    if uploaded_file.name.lower().endswith(".pdf") and size_kb > 1024:
        issues.append("PDF is large; consider uploading a single clear page")
    return issues


def validate_form(form):
    issues = []
    if not form["citizenship_number"].strip():
        issues.append("Missing citizenship number")
    if not form["full_name"].strip():
        issues.append("Missing full name")
    if not form["dob"].strip():
        issues.append("Missing date of birth")
    if not form["district"].strip():
        issues.append("Missing district")
    if not form["municipality"].strip():
        issues.append("Missing municipality")
    if not form["ward_number"].strip():
        issues.append("Missing ward number")
    return issues


def render_stepper():
    st.sidebar.subheader("Stepper")
    for index, step in enumerate(STEPS):
        if index < st.session_state.step:
            marker = "[x]"
        elif index == st.session_state.step:
            marker = "[>]"
        else:
            marker = "[ ]"
        st.sidebar.write(f"{marker} {step}")


init_state()

st.title("Smart KYC: Clarity at Every Step.")
st.caption("Intelligent KYC Experience - Smart Form Assistance and Support Ticket Reduction")
st.divider()

st.sidebar.title("Smart KYC Demo")
render_stepper()

selected_step = st.sidebar.selectbox("Jump to step", STEPS, index=st.session_state.step)
if STEPS.index(selected_step) != st.session_state.step:
    st.session_state.step = STEPS.index(selected_step)

st.sidebar.write(f"Current status: {st.session_state.status}")
st.sidebar.progress((st.session_state.step + 1) / len(STEPS))

form = st.session_state.form

if st.session_state.step == 0:
    st.subheader("Document Upload and KYC Form")
    st.write("Upload the citizenship certificate first to auto-fill the form.")
    uploaded = st.file_uploader("Upload citizenship certificate", type=["png", "jpg", "jpeg", "pdf"])
    st.caption("OCR uses EasyOCR locally; first run can take longer.")
    if uploaded:
        enhance_image = st.checkbox("Enhance image for OCR (recommended for noisy scans)", value=True)
        st.write(f"File: {uploaded.name} ({uploaded.size / 1024:.0f} KB)")
        issues = quality_issues(uploaded)
        if issues:
            st.warning("Quality checks found issues:")
            for issue in issues:
                st.write(f"- {issue}")
        else:
            st.success("Quality checks passed")

        file_bytes = uploaded.getvalue()
        file_ext = os.path.splitext(uploaded.name)[1].lower()
        with st.spinner("Running OCR..."):
            try:
                text_lines, avg_conf, ocr_results = extract_text_lines(file_bytes, file_ext, enhance_image)
            except Exception as exc:
                st.error(f"OCR failed: {exc}")
                text_lines, avg_conf, ocr_results = [], 0.0, []

        if avg_conf and avg_conf < 0.4:
            st.warning("OCR confidence is low; results may be inaccurate. Try a clearer scan.")

        st.session_state.ocr_data = parse_citizenship_fields(text_lines, ocr_results)
        st.session_state.ocr_file_name = uploaded.name
        if st.session_state.ocr_data:
            if st.session_state.ocr_applied_for != uploaded.name:
                apply_ocr_to_form(form, st.session_state.ocr_data, overwrite=True)
                st.session_state.ocr_applied_for = uploaded.name
                st.success("Form auto-filled from document")
        else:
            st.warning("OCR ran, but fields could not be mapped. Please fill manually.")

        st.subheader("OCR extraction (EasyOCR)")
        if text_lines:
            with st.expander("View extracted text"):
                st.write("\n".join(text_lines))
        st.json(st.session_state.ocr_data)
        if st.session_state.ocr_data and st.button("Reapply OCR values"):
            apply_ocr_to_form(form, st.session_state.ocr_data, overwrite=True)
            st.success("OCR values applied to form")
    else:
        st.info("Upload a document to auto-fill the form fields.")

    st.divider()
    col1, col2 = st.columns(2)
    form["citizenship_number"] = col1.text_input(
        "Citizenship number",
        value=form["citizenship_number"],
    )
    form["full_name"] = col1.text_input("Full name", value=form["full_name"])
    form["municipality"] = col1.text_input("Municipality", value=form["municipality"])
    form["dob"] = col2.text_input("Date of birth (YYYY-MM-DD)", value=form["dob"])
    form["district"] = col2.text_input("District", value=form["district"])
    form["ward_number"] = col2.text_input("Ward number", value=form["ward_number"])
    st.info("Extracted values can be edited before submission.")

if st.session_state.step == 1:
    st.subheader("Review and Submit")
    st.write("Confirm details before submission.")
    st.table(
        {
            "Field": list(form.keys()),
            "Value": list(form.values()),
        }
    )
    issues = validate_form(form)
    if issues:
        st.warning("Please fix these items before submission:")
        for issue in issues:
            st.write(f"- {issue}")
    else:
        st.success("All checks passed")

    consent = st.checkbox("I confirm the data is accurate")
    submit_disabled = bool(issues) or not consent
    if st.button("Submit for review", disabled=submit_disabled):
        st.session_state.status = "In Review"
        st.session_state.step = 2

if st.session_state.step == 2:
    st.subheader("Status Tracker")
    status = st.session_state.status
    st.write(f"Current status: {status}")

    progress_map = {
        "Draft": 0.2,
        "In Review": 0.6,
        "Approved": 1.0,
        "Rejected": 1.0,
    }
    st.progress(progress_map.get(status, 0.2))

    st.write("Timeline")
    timeline = [
        ("Submitted", "Documents received"),
        ("In Review", "Verification in progress"),
        ("Decision", "Approved or Rejected"),
    ]
    for label, detail in timeline:
        marker = "[x]" if status in ["In Review", "Approved", "Rejected"] or label == "Submitted" else "[ ]"
        if status in ["Approved", "Rejected"] and label == "Decision":
            marker = "[x]"
        st.write(f"{marker} {label} - {detail}")

    st.divider()
    st.subheader("Contextual Help")
    st.write("Actionable hints reduce support tickets and resubmissions.")
    st.info("Example: Photo is blurry - retake in good light with all edges visible.")
    st.info("Example: Date of birth mismatch - ensure it matches your ID document.")

    if status == "Rejected":
        st.warning("Rejection reasons")
        st.write("- Document photo is cropped")
        st.write("- Address does not match ID")
        st.write("Suggested actions are shown inline to fix and resubmit.")

    st.divider()
    st.subheader("Support Ticket Reduction")
    col1, col2, col3 = st.columns(3)
    col1.metric("Estimated error reduction", "35%")
    col2.metric("Estimated ticket reduction", "25%")
    col3.metric("Avg. resubmission time", "-40%")

    st.write("Self-service help")
    with st.expander("Common questions"):
        st.write("- Why is my status still in review? Typical review time is 1-2 days.")
        st.write("- How do I fix a mismatch? Use the guided checklist above.")
        st.write("- What documents are accepted? Citizenship ID, passport, or driving license.")

    st.divider()
    st.subheader("Simulate outcomes")
    col1, col2, col3 = st.columns(3)
    if col1.button("Mark Approved"):
        st.session_state.status = "Approved"
    if col2.button("Mark Rejected"):
        st.session_state.status = "Rejected"
    if col3.button("Reset to In Review"):
        st.session_state.status = "In Review"

st.divider()
nav_left, nav_right, _ = st.columns([1, 1, 6])
with nav_left:
    if st.button("Back", disabled=st.session_state.step == 0):
        st.session_state.step = max(0, st.session_state.step - 1)
with nav_right:
    disable_next = st.session_state.step >= len(STEPS) - 1 or st.session_state.step == 1
    if st.button("Next", disabled=disable_next):
        st.session_state.step = min(len(STEPS) - 1, st.session_state.step + 1)
