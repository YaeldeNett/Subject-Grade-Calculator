# Subject-Grade-Calculator
A web-based Uni Grade Calculator built with Streamlit. Lets you add subjects, assignments, exams, and marks (percentages or fractions like 14/20) to see your current contribution and the average you need on remaining assessments to pass. Supports loading/saving JSON across multiple subjects.  Vibe-coded with the help of GPT-5

# Uni Grade Calculator — JSON Load/Save + Multi‑subject (Streamlit)

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Use
- **Load** your `grades.json` via the sidebar file uploader.
- Pick a subject, edit rows (Mark can be `75` or `14/20`).
- **Download JSON (all subjects)** to save everything, or **Download CSV (current subject)**.
