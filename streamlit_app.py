# Streamlit Uni Grade Calculator — Load/Save JSON + Multi‑subject (Compat)
# Save as: streamlit_app.py
# Run:
#   pip install -r requirements.txt
#   streamlit run streamlit_app.py
#
# JSON schema supported (example):
# {
#   "Math2": {
#     "title": "Math2",
#     "assessments": [
#       {"name": "M1", "kind": "Quiz", "weight": 5.0, "mark": 70.0},
#       ...
#     ]
#   },
#   "Data Analytics": { ... }
# }
#
# - Load: use the file uploader (left sidebar).
# - Edit: pick a subject, change rows/marks (mark can be "75" or "14/20").
# - Save: buttons to download JSON (all subjects) or CSV (current subject).

import io
import json
import math
from typing import Optional, Dict, Any

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Uni Grade Calculator", layout="centered")

# ---------- Helpers ----------

def parse_mark(x: str) -> Optional[float]:
    """Return the mark as a percentage in [0, 100] if possible, else None."""
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    try:
        if "/" in s:
            a, b = s.split("/", 1)
            a = float(a.strip()); b = float(b.strip())
            if b <= 0:
                return None
            return max(0.0, min((a / b) * 100.0, 100.0))
        # plain number interpreted as a percentage already
        return max(0.0, min(float(s), 100.0))
    except Exception:
        return None

def editor(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Backwards compatible editable table."""
    ed = getattr(st, "data_editor", None) or getattr(st, "experimental_data_editor", None)
    if ed is None:
        st.warning("Your Streamlit is very old; showing read‑only table. Upgrade Streamlit for editing.")
        st.dataframe(df)
        return df.copy()
    # Try with nicer args; fall back if TypeError on older versions
    try:
        return ed(df, num_rows="dynamic", use_container_width=True, key=key)
    except TypeError:
        return ed(df, key=key)

def df_from_subject(subject: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for a in subject.get("assessments", []):
        rows.append({
            "Name": a.get("name", ""),
            "Type": a.get("kind", ""),
            "Weight %": a.get("weight", 0.0),
            "Mark": "" if a.get("mark", None) is None else a.get("mark"),
        })
    if not rows:
        rows = [{"Name":"","Type":"Assessment","Weight %":0.0,"Mark":""}]
    return pd.DataFrame(rows, columns=["Name","Type","Weight %","Mark"])

def subject_from_df(title: str, df: pd.DataFrame) -> Dict[str, Any]:
    subj = {"title": title, "assessments": []}
    for _, r in df.iterrows():
        name = str(r.get("Name", "")).strip()
        kind = str(r.get("Type", "Assessment")).strip() or "Assessment"
        weight = float(pd.to_numeric(r.get("Weight %", 0.0), errors="coerce") or 0.0)
        # store mark as number if parseable, else null
        parsed = parse_mark(r.get("Mark"))
        mark = None if parsed is None else float(parsed)
        subj["assessments"].append({
            "name": name, "kind": kind, "weight": weight, "mark": mark
        })
    return subj

def compute_stats(df: pd.DataFrame, pass_mark: float):
    weights = pd.to_numeric(df["Weight %"], errors="coerce").fillna(0.0)
    marks = df["Mark"].apply(parse_mark)
    completed_mask = marks.notna()
    completed_weight = float(weights[completed_mask].sum())
    contribution = float((weights[completed_mask] * (marks[completed_mask] / 100.0)).sum())
    planned_weight = float(weights.sum())
    remaining_planned_weight = max(0.0, 100.0 - completed_weight)
    current_avg_completed = (contribution / completed_weight) * 100.0 if completed_weight > 1e-9 else None
    if remaining_planned_weight <= 1e-9:
        needed_avg_remaining = 0.0 if contribution >= pass_mark else math.inf
    else:
        needed_avg_remaining = (pass_mark - contribution) / (remaining_planned_weight / 100.0)
        needed_avg_remaining = max(0.0, min(needed_avg_remaining, 9999.0))
    return {
        "completed_weight": completed_weight,
        "contribution": contribution,
        "planned_weight": planned_weight,
        "remaining_planned_weight": remaining_planned_weight,
        "current_avg_completed": current_avg_completed,
        "needed_avg_remaining": needed_avg_remaining,
    }

# ---------- Sidebar: Load / Settings ----------

st.title("Uni Grade Calculator (Web)")
st.caption("Load/save JSON, switch subjects, edit rows. Mark accepts 75 or 14/20.")

with st.sidebar:
    st.header("Load / Save")
    uploaded = st.file_uploader("Load JSON file", type=["json"], accept_multiple_files=False)
    PASS_MARK = st.number_input("Pass mark (%)", 0.0, 100.0, 50.0, 1.0)

# Session state for the whole book of subjects
if "book" not in st.session_state:
    # default book with one sample subject
    st.session_state.book = {
        "Sample Subject": {
            "title": "Sample Subject",
            "assessments": [
                {"name":"Assignment 1","kind":"Assignment","weight":20.0,"mark":75.0},
                {"name":"Midterm","kind":"Exam","weight":30.0,"mark":None},
                {"name":"Final","kind":"Exam","weight":50.0,"mark":None},
            ]
        }
    }
if "current" not in st.session_state:
    st.session_state.current = "Sample Subject"

# If a file is uploaded, load it into session
if uploaded is not None:
    try:
        data = json.load(uploaded)
        if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
            st.session_state.book = data
            # pick first subject
            st.session_state.current = list(data.keys())[0] if data else "New Subject"
            st.success("JSON loaded into the app.")
        else:
            st.error("Invalid JSON format. Expected a dict of subjects → {title, assessments:[...]}." )
    except Exception as e:
        st.error(f"Failed to parse JSON: {e}")

# ---------- Subject selector / add-delete ----------

left, right = st.columns([2,1])
with left:
    subject_names = list(st.session_state.book.keys())
    current = st.selectbox("Subject", subject_names + ["+ Add new subject..."], index=subject_names.index(st.session_state.current) if st.session_state.current in subject_names else 0)
with right:
    if st.button("Delete subject", help="Remove the current subject") and current in st.session_state.book and len(st.session_state.book) > 1:
        del st.session_state.book[current]
        st.session_state.current = list(st.session_state.book.keys())[0]
        st.rerun()

if current == "+ Add new subject...":
    new_name = st.text_input("New subject name", value="New Subject")
    if st.button("Create subject"):
        if new_name.strip() == "" or new_name in st.session_state.book:
            st.warning("Choose a unique non‑empty name.")
        else:
            st.session_state.book[new_name] = {"title": new_name, "assessments": []}
            st.session_state.current = new_name
            st.rerun()
else:
    st.session_state.current = current

# ---------- Data editor for current subject ----------

subject = st.session_state.book[st.session_state.current]
df = df_from_subject(subject)
df = editor(df, key="table")

# Guarantee expected columns and order
for c in ["Name","Type","Weight %","Mark"]:
    if c not in df.columns:
        df[c] = "" if c != "Weight %" else 0.0
df = df[["Name","Type","Weight %","Mark"]]

# ---------- Stats ----------

stats = compute_stats(df, PASS_MARK)
st.subheader("Subject summary")
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Completed weight", f"{stats['completed_weight']:.2f}%")
    st.metric("Planned weight", f"{stats['planned_weight']:.2f}%")
with m2:
    st.metric("Contribution so far", f"{stats['contribution']:.2f}%")
    st.metric("Remaining planned weight", f"{stats['remaining_planned_weight']:.2f}%")
with m3:
    st.metric("Current avg on completed", "—" if stats['current_avg_completed'] is None else f"{stats['current_avg_completed']:.2f}%")
    st.metric("Required avg on remaining to pass",
              "Impossible (no remaining weight)" if stats['needed_avg_remaining'] == math.inf else f"{stats['needed_avg_remaining']:.2f}%")

# ---------- Save back to session ----------

# Update the current subject from edited DF
st.session_state.book[st.session_state.current] = subject_from_df(st.session_state.current, df)

st.divider()
st.markdown("### Save / Export")


# 1) Download full JSON (all subjects)
json_bytes = json.dumps(st.session_state.book, indent=2).encode("utf-8")
st.download_button(
    "Download JSON (all subjects)",
    data=json_bytes,
    file_name="grades.json",
    mime="application/json",
    help="Saves every subject in the app to a single JSON file."
)

# 2) Download current subject as CSV
csv_buf = io.StringIO()
df.to_csv(csv_buf, index=False)
st.download_button(
    "Download CSV (current subject)",
    data=csv_buf.getvalue().encode("utf-8"),
    file_name=f"{st.session_state.current.replace(' ','_')}.csv",
    mime="text/csv",
    help="Saves just the currently selected subject as a CSV."
)

st.divider()
st.write(":bulb: **Tips**") 
st.write("- You can load your existing JSON, edit, then re‑download.")
st.write("- Marks accept `75` or `14/20`. Saved JSON stores marks as numbers (percent) or null if blank.")
st.write("- Add multiple subjects and switch between them via the dropdown above.")
