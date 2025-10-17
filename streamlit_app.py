# Streamlit Uni Grade Calculator — reliable editing, live fraction→percent conversion, thick stacked bar
# Save as: streamlit_app.py

import io
import json
import math
from typing import Optional, Dict, Any

import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Uni Grade Calculator (Web)", layout="centered")

# ---------- Helpers ----------

def parse_mark(x) -> Optional[float]:
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
        return max(0.0, min(float(s), 100.0))
    except Exception:
        return None

def format_pct_or_empty(val: Optional[float]) -> str:
    return "" if val is None else f"{float(val):.1f}"

def normalize_marks_in_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'a/b' or numeric strings to a standard percentage string with one decimal."""
    df2 = df.copy()
    if "Mark" in df2.columns:
        df2["Mark"] = df2["Mark"].apply(parse_mark).apply(format_pct_or_empty).astype(object)
    return df2

def editor(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Editable table with explicit column config to prevent type glitches."""
    ed = getattr(st, "data_editor", None) or getattr(st, "experimental_data_editor", None)
    if ed is None:
        st.warning("Old Streamlit detected; showing read-only table.")
        st.dataframe(df)
        return df.copy()

    df = df.copy()
    if "Mark" in df.columns:
        df["Mark"] = df["Mark"].astype(object).fillna("")
    if "Weight %" in df.columns:
        df["Weight %"] = pd.to_numeric(df["Weight %"], errors="coerce")

    col_config = {
        "Name": st.column_config.TextColumn("Name", width="medium", help="Assessment name"),
        "Type": st.column_config.TextColumn("Type", help="Quiz / Exam / Assignment etc."),
        "Weight %": st.column_config.NumberColumn(
            "Weight %", help="Percent weight of this assessment",
            min_value=0.0, max_value=100.0, step=0.5, format="%.1f"
        ),
        "Mark": st.column_config.TextColumn(
            "Mark", help="Enter 75 or 2/5. Auto-converts to a percentage."
        ),
    }

    try:
        return ed(
            df,
            num_rows="dynamic",
            use_container_width=True,
            column_config=col_config,
            key=key
        )
    except TypeError:
        return ed(df, key=key)

def df_from_subject(subject: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for a in subject.get("assessments", []):
        rows.append({
            "Name": a.get("name", ""),
            "Type": a.get("kind", ""),
            "Weight %": a.get("weight", 0.0),
            "Mark": format_pct_or_empty(a.get("mark", None)),
        })
    if not rows:
        rows = [{"Name":"","Type":"Assessment","Weight %":0.0,"Mark":""}]
    df = pd.DataFrame(rows, columns=["Name","Type","Weight %","Mark"])
    df["Mark"] = df["Mark"].astype(object)
    return df

def subject_from_df(title: str, df: pd.DataFrame) -> Dict[str, Any]:
    subj = {"title": title, "assessments": []}
    for _, r in df.iterrows():
        subj["assessments"].append({
            "name": str(r.get("Name", "")).strip(),
            "kind": (str(r.get("Type", "Assessment")).strip() or "Assessment"),
            "weight": float(pd.to_numeric(r.get("Weight %", 0.0), errors="coerce") or 0.0),
            "mark": (lambda v: None if v is None else float(v))(parse_mark(r.get("Mark")))
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
    needed_avg_remaining = (
        0.0 if remaining_planned_weight <= 1e-9 and contribution >= pass_mark
        else (math.inf if remaining_planned_weight <= 1e-9
              else max(0.0, min((pass_mark - contribution) / (remaining_planned_weight / 100.0), 9999.0)))
    )
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
st.caption("Load/save JSON, switch subjects, edit rows. Mark accepts 75 or 2/5 (auto-converts to percent).")

with st.sidebar:
    st.header("Load / Save")
    uploaded = st.file_uploader("Load JSON file", type=["json"], accept_multiple_files=False)
    PASS_MARK = st.number_input("Pass mark (%)", 0.0, 100.0, 50.0, 1.0)

# Session state stores book (data), current subject key, and per-subject tables
if "book" not in st.session_state:
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
if "tables" not in st.session_state:
    st.session_state.tables = {k: df_from_subject(v) for k, v in st.session_state.book.items()}

# Load JSON (replaces everything)
if uploaded is not None:
    try:
        data = json.load(uploaded)
        if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
            st.session_state.book = data
            st.session_state.current = list(data.keys())[0] if data else "New Subject"
            st.session_state.tables = {k: df_from_subject(v) for k, v in data.items()}
            st.success("JSON loaded.")
        else:
            st.error("Invalid JSON format. Expected a dict of subjects → {title, assessments:[...]}." )
    except Exception as e:
        st.error(f"Failed to parse JSON: {e}")

# ---------- Subject selector / add-delete ----------

left, right = st.columns([2,1])
with left:
    subject_names = list(st.session_state.book.keys())
    idx = subject_names.index(st.session_state.current) if st.session_state.current in subject_names else 0
    current = st.selectbox("Subject", subject_names + ["+ Add new subject..."], index=idx)
with right:
    if st.button("Delete subject", help="Remove the current subject") and current in st.session_state.book and len(st.session_state.book) > 1:
        del st.session_state.book[current]
        st.session_state.tables.pop(current, None)
        st.session_state.current = list(st.session_state.book.keys())[0]
        st.rerun()

if current == "+ Add new subject...":
    new_name = st.text_input("New subject name", value="New Subject")
    if st.button("Create subject"):
        if new_name.strip() == "" or new_name in st.session_state.book:
            st.warning("Choose a unique non-empty name.")
        else:
            st.session_state.book[new_name] = {"title": new_name, "assessments": []}
            st.session_state.tables[new_name] = df_from_subject(st.session_state.book[new_name])
            st.session_state.current = new_name
            st.rerun()
else:
    st.session_state.current = current

# ---------- Data editor for current subject ----------

subject_key = st.session_state.current
subject = st.session_state.book[subject_key]
df_state = st.session_state.tables.get(subject_key, df_from_subject(subject))

# 1) Show editor (unique key per subject)
df_edit = editor(df_state, key=f"table_{subject_key}")

# 2) Immediately persist any table edits (so metrics/graph stay in sync even if only weights changed)
st.session_state.tables[subject_key] = df_edit.copy()

# 3) Normalize any fraction / numeric text immediately and refresh if changed
df_norm = normalize_marks_in_df(df_edit)
if not df_norm.equals(df_edit):
    st.session_state.tables[subject_key] = df_norm
    st.rerun()

# 4) Use the normalized dataframe for calculations & display
df = st.session_state.tables[subject_key].copy()

# Guarantee expected columns and order
for c in ["Name","Type","Weight %","Mark"]:
    if c not in df.columns:
        df[c] = "" if c != "Weight %" else 0.0
df = df[["Name","Type","Weight %","Mark"]]

# ---------- Stats (always reflect current subject and latest edits) ----------

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

# ---------- Save back to session (keeps JSON export up-to-date) ----------

st.session_state.book[subject_key] = subject_from_df(subject_key, df)

# ---------- Progress chart (per subject) ----------

st.divider()
st.markdown("### Progress Overview (this subject)")

# Compute three segments to display (values sum to 100 when planned=100)
contrib = round(stats["contribution"], 1)
completed_but_not_contrib = max(0.0, round(stats["completed_weight"] - stats["contribution"], 1))
remaining = max(0.0, round(stats["remaining_planned_weight"], 1))

chart_df = pd.DataFrame({
    "Row": ["All","All","All"],
    "Segment": ["Contribution so far", "Completed weight", "Planned Weight"],
    "Value": [contrib, completed_but_not_contrib, remaining],
    "Label": [f"{contrib:.1f}%", f"{completed_but_not_contrib:.1f}%", f"{remaining:.1f}%"]
})

# Thick single-row stacked bar
bar = alt.Chart(chart_df).mark_bar(size=50).encode(
    y=alt.Y("Row:N", axis=None),
    x=alt.X("Value:Q", stack="zero",
            axis=alt.Axis(title=None, labels=False, ticks=False),
            scale=alt.Scale(domain=[0, 100])),
    color=alt.Color("Segment:N", legend=alt.Legend(orient="top"))
).properties(height=120, width=900)

labels = alt.Chart(chart_df).mark_text(baseline="middle").encode(
    y=alt.Y("Row:N"),
    x=alt.X("Value:Q", stack="center"),
    text="Label",
    color=alt.value("white")
)

st.altair_chart(bar + labels, use_container_width=True)

# ---------- Save / Export ----------

st.divider()
st.markdown("### Save / Export")

json_bytes = json.dumps(st.session_state.book, indent=2).encode("utf-8")
st.download_button(
    "Download JSON (all subjects)",
    data=json_bytes,
    file_name="grades.json",
    mime="application/json"
)

csv_buf = io.StringIO()
df.to_csv(csv_buf, index=False)
st.download_button(
    "Download CSV (current subject)",
    data=csv_buf.getvalue().encode("utf-8"),
    file_name=f"{subject_key.replace(' ','_')}.csv",
    mime="text/csv"
)

st.divider()
st.write(":bulb: **Tips**")
st.write("- Marks accept `75` or `2/5`. They auto-convert to a percentage with one decimal place.")
st.write("- The stacked bar and metrics always reflect the table above for the selected subject.")
