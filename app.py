# AI Resume Screening System - Day 7
# Replace your app.py with this file.

import streamlit as st
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Screening System",
                   page_icon="🤖",
                   layout="wide",
                   initial_sidebar_state="expanded")

with st.sidebar:
    st.title("🤖 AI Resume Screening")
    st.markdown("---")
    st.write("### Features")
    for item in [
        "Resume Upload","AI Resume Analysis","ATS Score",
        "Skill Matching","Interview Recommendation","Download Report"
    ]:
        st.write("✅", item)
    st.markdown("---")
    st.info("Minor Project\nB.Tech CSE (AI)")

skills=[
"Python","Java","C","C++","SQL","MySQL","Machine Learning","Deep Learning",
"Data Analysis","Pandas","NumPy","TensorFlow","Flask","Streamlit","HTML",
"CSS","JavaScript","Git","Communication","Team Work","Problem Solving",
"Leadership","Analytical Thinking","Database","RFID"
]

st.title("🤖 AI Resume Screening System")
st.success("Welcome! Upload your resume and compare it with the job description using AI.")

left,right=st.columns(2)
with left:
    uploaded_file=st.file_uploader("📄 Upload Resume (PDF)",type=["pdf"])
with right:
    job_description=st.text_area("📝 Enter Job Description",height=220)

c1,c2,c3=st.columns([1,2,1])
with c2:
    analyze=st.button("🚀 Analyze Resume",use_container_width=True)

if analyze:
    if not uploaded_file:
        st.warning("Please upload a resume.")
        st.stop()
    if not job_description.strip():
        st.warning("Please enter a job description.")
        st.stop()

    resume_text=""
    with pdfplumber.open(uploaded_file) as pdf:
        for p in pdf.pages:
            t=p.extract_text()
            if t:
                resume_text+=t+"\n"

    name=resume_text.split("\n")[0] if resume_text else "Not Found"
    emails=re.findall(r'[\w\.-]+@[\w\.-]+',resume_text)
    email=emails[0] if emails else "Not Found"
    phones=re.findall(r'\+?\d[\d\s-]{8,15}',resume_text)
    phone=phones[0] if phones else "Not Found"

    tfidf=TfidfVectorizer().fit_transform([job_description,resume_text])
    score=cosine_similarity(tfidf[0:1],tfidf[1:2])[0][0]*100
    ats=min(score+5,100)

    matched=[s for s in skills if s.lower() in resume_text.lower()]
    missing=[s for s in skills if s.lower() not in resume_text.lower()]

    st.divider()
    st.subheader("👤 Candidate Information")
    a,b,c=st.columns(3)
    a.metric("Name",name)
    b.metric("Email",email)
    c.metric("Phone",phone)

    st.divider()
    x,y=st.columns(2)
    with x:
        st.subheader("📊 Resume Match Score")
        st.metric("Match Score",f"{score:.2f}%")
        st.progress(min(int(score),100))
    with y:
        st.subheader("⭐ ATS Score")
        st.metric("ATS",f"{ats:.2f}%")
        if score>=80:
            st.success("★★★★★ Excellent")
        elif score>=60:
            st.info("★★★★ Good")
        elif score>=40:
            st.warning("★★★ Average")
        else:
            st.error("★★ Needs Improvement")

    st.divider()
    m1,m2=st.columns(2)
    with m1:
        st.subheader("✅ Matched Skills")
        for s in matched: st.write("✔️",s)
    with m2:
        st.subheader("❌ Missing Skills")
        for s in missing: st.write("✖️",s)

    st.divider()
    st.subheader("📚 Suggested Skills")
    for s in missing[:5]:
        st.write("📌",s)

    st.divider()
    st.subheader("🎯 Recommendation")
    if score>=80:
        st.balloons()
        st.success("Excellent Match! Recommended for Interview.")
    elif score>=60:
        st.info("Good Match. Can Be Considered.")
    elif score>=40:
        st.warning("Average Match. Improve your skills.")
    else:
        st.error("Low Match.")

    st.divider()
    st.subheader("📄 Resume Statistics")
    s1,s2,s3=st.columns(3)
    s1.metric("Words",len(resume_text.split()))
    s2.metric("Matched Skills",len(matched))
    s3.metric("Missing Skills",len(missing))

    report=f"""AI Resume Screening Report

Name: {name}
Email: {email}
Phone: {phone}

Resume Score: {score:.2f}%
ATS Score: {ats:.2f}%

Matched Skills:
{', '.join(matched)}

Missing Skills:
{', '.join(missing)}
"""
    st.download_button("📥 Download Report",report,"Resume_Report.txt","text/plain")

    with st.expander("📃 Resume Preview"):
        st.text(resume_text)

st.divider()
st.caption("Developed using Python, Streamlit, NLP, TF-IDF and Cosine Similarity.")
