import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Student Career Prediction", page_icon="🎓", layout="wide")
st.title("🎓 Student Career Prediction: Eligibility & Package Estimator")

st.markdown(
    """
This app lets you **upload a dataset** (or use the sample) and then enter **a new student's details** to:
1. Predict **Placement Eligibility** (Yes/No + probability)  
2. Estimate **Package (LPA)** if eligible  
3. Provide **actionable recommendations** if not eligible  

**Dataset schema expected:**
`Student_Name, Roll_Number, Gender, Age, 10th_Percentage, 12th_Percentage, UG_CGPA, Backlogs_Count, Attendance_%, Technical_Skill_Score, Primary_Tech, Soft_Skill_Score, Soft_Skill_Strong_Area, Programming_Languages_Known, Projects_Count, Internships_Count, Certifications_Count, Study_Hours_per_Day, Preferred_Domain, Career_Path, Placement_Eligible, Package_LPA`
"""
)

# Load sample dataset
@st.cache_data(show_spinner=False)
def load_sample():
    df = pd.read_csv("student_career_dataset.csv")  # fixed path
    return df

# Sidebar: dataset upload
with st.sidebar:
    st.header("📂 Dataset")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])   
    use_sample = st.checkbox("Use sample dataset", value=True if not uploaded else False)

    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
    elif use_sample:
        df_raw = load_sample()
    else:
        st.stop()

    st.success(f"Dataset loaded with {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

# ----- Modeling utils -----
label_col = "Placement_Eligible"  # Yes/No
salary_col = "Package_LPA"        # numeric (train on eligible only)

num_cols = [
    "Age","10th_Percentage","12th_Percentage","UG_CGPA","Backlogs_Count","Attendance_%",
    "Technical_Skill_Score","Soft_Skill_Score","Programming_Languages_Known","Projects_Count",
    "Internships_Count","Certifications_Count","Study_Hours_per_Day"
]

cat_cols = [
    "Gender","Primary_Tech","Soft_Skill_Strong_Area","Preferred_Domain","Career_Path"
]

# Clean/align types
for c in num_cols:
    if c in df_raw.columns:
        df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

for c in cat_cols + [label_col]:
    if c in df_raw.columns:
        df_raw[c] = df_raw[c].astype(str)

# Drop rows with missing critical fields
req_cols = num_cols + cat_cols + [label_col, salary_col]
df = df_raw.dropna(subset=[c for c in req_cols if c in df_raw.columns]).copy()

# Convert labels to 0/1
if label_col in df.columns:
    df["eligible_bin"] = df[label_col].str.lower().map({"yes":1, "no":0})
else:
    st.error("Dataset missing 'Placement_Eligible' column.")
    st.stop()

X = df[num_cols + cat_cols]
y = df["eligible_bin"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

clf = Pipeline([
    ("prep", preprocess),
    ("model", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)

# Salary model (only train on eligible rows)
eligible_rows = df[df["eligible_bin"] == 1]
if eligible_rows.shape[0] > 10:
    Xsal = eligible_rows[num_cols + cat_cols]
    ysal = eligible_rows[salary_col].astype(float)

    reg = Pipeline([
        ("prep", preprocess),
        ("model", RandomForestRegressor(n_estimators=300, random_state=42))
    ])
    reg.fit(Xsal, ysal)
else:
    reg = None

st.info(f"Eligibility model accuracy on holdout: **{acc:.2%}**")

# ---------- New Student Form ----------
st.subheader("🧑‍🎓 Enter New Student Details")
with st.form("new_student"):
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Male","Female","Other"]) 
        age = st.number_input("Age", 16, 40, 21)
        perc10 = st.number_input("10th_Percentage", 0.0, 100.0, 78.0, step=0.1)
        perc12 = st.number_input("12th_Percentage", 0.0, 100.0, 80.0, step=0.1)
        cgpa = st.number_input("UG_CGPA", 0.0, 10.0, 7.5, step=0.1)
        backlogs = st.number_input("Backlogs_Count", 0, 20, 0)
    with c2:
        attendance = st.number_input("Attendance_%", 0.0, 100.0, 85.0, step=0.1)
        tech = st.number_input("Technical_Skill_Score", 0.0, 100.0, 70.0, step=1.0)
        primary_tech = st.selectbox("Primary_Tech", ["C","Python","Java","JavaScript","Other"]) 
        soft = st.number_input("Soft_Skill_Score", 0.0, 100.0, 68.0, step=1.0)
        soft_area = st.selectbox("Soft_Skill_Strong_Area", ["Communication","Project Management","Critical Thinking","Problem-Solving","Leadership"]) 
        langs = st.number_input("Programming_Languages_Known", 1, 20, 4)
    with c3:
        projects = st.number_input("Projects_Count", 0, 50, 3)
        internships = st.number_input("Internships_Count", 0, 10, 1)
        certs = st.number_input("Certifications_Count", 0, 50, 2)
        study = st.number_input("Study_Hours_per_Day", 0.0, 16.0, 3.0, step=0.1)
        domain = st.selectbox("Preferred_Domain", ["Web Development","Data Science","AI/ML","Cybersecurity","Cloud Computing","Mobile Development"]) 
        career = st.selectbox("Career_Path", [
            "Software Developer","Web Developer","Frontend Developer","Backend Developer",
            "Data Analyst","Data Scientist","ML Engineer","Security Analyst","SOC Engineer",
            "Network Engineer","Cloud Engineer","DevOps Engineer","SRE","Android Developer",
            "iOS Developer","Mobile QA"
        ])

    submitted = st.form_submit_button("Predict")

if submitted:
    inp = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "10th_Percentage": perc10,
        "12th_Percentage": perc12,
        "UG_CGPA": cgpa,
        "Backlogs_Count": backlogs,
        "Attendance_%": attendance,
        "Technical_Skill_Score": tech,
        "Primary_Tech": primary_tech,
        "Soft_Skill_Score": soft,
        "Soft_Skill_Strong_Area": soft_area,
        "Programming_Languages_Known": langs,
        "Projects_Count": projects,
        "Internships_Count": internships,
        "Certifications_Count": certs,
        "Study_Hours_per_Day": study,
        "Preferred_Domain": domain,
        "Career_Path": career
    }])

    proba = clf.predict_proba(inp)[0,1]
    eligible_pred = (proba >= 0.5)

    if eligible_pred:
        st.success(f"✅ Eligible for placement drive (confidence: {proba:.1%})")
        if reg is not None:
            est_salary = float(reg.predict(inp)[0])
            st.metric("Estimated Package (LPA)", f"{est_salary:.2f}")
        else:
            st.info("Salary model not trained (not enough eligible rows in dataset).")
    else:
        st.error(f"❌ Not eligible yet (confidence: {1-proba:.1%} against)")

        # Rule-based recommendations
        recs = []
        def need(threshold, actual, msg):
            if actual < threshold:
                recs.append(msg.format(gap=threshold-actual))
        need(7.0, cgpa, f"Raise **UG_CGPA** to at least 7.0 (gap: {{gap:.2f}}).")
        need(75.0, attendance, f"Improve **Attendance** to 75%+ (gap: {{gap:.1f}}%).")
        if backlogs > 1:
            recs.append("Clear backlogs to **≤ 1**.")
        need(60.0, tech, f"Boost **Technical_Skill_Score** to 60+ (gap: {{gap:.0f}}).")
        need(55.0, soft, f"Boost **Soft_Skill_Score** to 55+ (gap: {{gap:.0f}}).")
        need(2, projects, f"Complete **at least 2 projects** (need {{gap:.0f}} more).")
        need(1, internships, f"Do **at least 1 internship** (need {{gap:.0f}}).")
        need(2.0, study, f"Increase **Study Hours/Day** to 2.0+ (gap: {{gap:.1f}}).")

        st.markdown("### 📌 Recommendations to become eligible")
        for r in recs:
            st.markdown(f"- {r}")
