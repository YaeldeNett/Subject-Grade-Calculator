# Streamlit Uni Grade Calculator — robust editor + live fraction→percent conversion
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
        # plain number interpreted as a percentage already
        return max(0.0, min(float(s), 100.0))
    except Exception:
        return None

def format_pct_or_empty(val: Optional[float]) -> str:
    return "" if val is None else f"{float(val):.1f}"

def normalize_marks_in_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any 'a/b' or numeric strings to a standard percentage string with one decimal."""
    df2 = df.copy()
    if "Mark" in df2.columns:
        df2["Mark"] = df2["Mark"].apply(parse_mark).apply(format_pct_or_empty)
        # Ensure object dtype (best compatibility for editing)
        df2["Mark"] = df2["Mark"].astype(object)
    return df2

def editor(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Editable table with explicit column config to prevent type glitches."""
    ed = getattr(st, "data_editor", None) or getattr(st, "experimental_data_editor", None)
    if ed is None:
        st.warning("Your Streamlit is very old; showing read-only table. Upgrade Streamlit for editing.")
        st.dataframe(df)
        return df.copy()

    # Ensure consistent dtypes before rendering
    df = df.copy()
    # Mark must be object/str so users can enter things like "14/20"
    if "Mark" in df.columns:
        df["Mark"] = df["Mark"].astype(object).fillna("")
    if "Weight %" in df.columns:
        df["Weight %"] = pd.to_numeric(df["Weight %"], errors="coerce")

    col_config = {
        "Name": st.column_config.TextColumn("Name", width="medium", help="Assessment name"),
        "Type": st.column_config.TextColumn("Type", help="Quiz / Exam / Assignment etc."),
        "Weight %": st.column_config.NumberColumn(
            "Weight %",
            help="Percent weight of this assessment",
            min_value=0.0, max_value=100.0, step=0.5, format="%.1f"
        ),
        "Mark": st.column_config.TextColumn(
            "Mark",
            help="Enter 75 or 14/20. It will auto-convert to a percentage."
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
    # Use object dtype for best editor compatibility
    df["Mark"] = df["Mark"].astype(object)
    return df

def subject_from_df(title: str, df: pd.DataFrame) -> Dict[str, Any]:
    subj = {"title": title, "assessments": []}
    for _, r in df.iterrows():
        name = str(r.get("Name", "")).strip()
        kind = str(r.get("Type", "Assessment")).strip() or "Assessment"
        weight = float(pd.to_numeric(r.get("Weight %", 0.0), errors="coerce") or 0.0)
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
st.caption("Load/save JSON, switch subjects, edit rows. Mark accepts 75 or 14/20 (auto-converts to percent).")

with st.sidebar:
    st.header("Load / Save")
    uploaded = st.file_uploader("Load JSON file", type=["json"], accept_multiple_files=False)
    PASS_MARK = st.number_input("Pass mark (%)", 0.0, 100.0, 50.0, 1.0)

# Session state for the whole book of subjects + per-subject table cache
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

# If a file is uploaded, load it into session
if uploaded is not None:
    try:
        data = json.load(uploaded)
        if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
            st.session_state.book = data
            st.session_state.current = list(data.keys())[0] if data else "New Subject"
            st.session_state.tables = {k: df_from_subject(v) for k, v in data.items()}
            st.success("JSON loaded into the app.")
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

subject = st.session_state.book[st.session_state.current]
df_state = st.session_state.tables.get(st.session_state.current, df_from_subject(subject))

# 1) Show editor
df_edit = editor(df_state, key=f"table_{st.session_state.current}")

# 2) Normalize any fraction / numeric text immediately and refresh if changed
df_norm = normalize_marks_in_df(df_edit)
if not df_norm.equals(df_edit):
    st.session_state.tables[st.session_state.current] = df_norm
    st.rerun()

# 3) Use the normalized dataframe for calculations/saving
df = df_norm

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

# Update the per-subject DataFrame cache
st.session_state.tables[st.session_state.current] = df.copy()
# Also update the structured book for JSON export
st.session_state.book[st.session_state.current] = subject_from_df(st.session_state.current, df)

# ---------- Progress chart (per subject, like your screenshot) ----------

st.divider()
st.markdown("### Progress Overview (this subject)")

contrib = round(stats["contribution"], 1)
completed_but_not_contrib = max(0.0, round(stats["completed_weight"] - stats["contribution"], 1))
remaining = max(0.0, round(stats["remaining_planned_weight"], 1))

chart_df = pd.DataFrame({
    "Segment": ["Contribution so far", "Completed weight", "Planned Weight"],
    "Value": [contrib, completed_but_not_contrib, remaining],
    "Label": [f"{contrib:.1f}%", f"{completed_but_not_contrib:.1f}%", f"{remaining:.1f}%"]
})

# Build a horizontal stacked bar to 100%
bar = alt.Chart(chart_df).mark_bar().encode(
    x=alt.X("sum(Value):Q", axis=alt.Axis(title=None, labels=False, ticks=False), scale=alt.Scale(domain=[0, 100])),
    y=alt.Y("N():Q", axis=None),  # fake single row
    color=alt.Color("Segment:N", legend=alt.Legend(orient="top")),
    order=alt.Order("Segment", sort="ascending"),
    tooltip=[alt.Tooltip("Segment:N"), alt.Tooltip("Value:Q", format=".1f")]
).properties(height=80, width=800)

labels = alt.Chart(chart_df).mark_text(baseline="middle", dy=0).encode(
    x=alt.X("sum(Value):Q", stack="center"),
    y=alt.Y("N():Q"),
    text="Label",
    color=alt.value("white")
)

st.altair_chart(bar + labels, use_container_width=True)

# ---------- Save / Export ----------

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
st.write("- Marks accept `75` or `14/20`. They auto-convert to a percentage with one decimal place.")
st.write("- If editing ever feels 'stuck', the subject switcher forces a refresh; but it should be smooth now.")
