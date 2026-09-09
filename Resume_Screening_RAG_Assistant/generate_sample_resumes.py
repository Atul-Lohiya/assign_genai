"""
Generates three sample resume PDFs (Resume_A, Resume_B, Resume_C) used to
demo the AI Resume Screening Assistant against sample_data/jd_data_scientist.txt.

Run once:  python generate_sample_resumes.py
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

styles = getSampleStyleSheet()
h1 = ParagraphStyle("h1", parent=styles["Heading1"], spaceAfter=4)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], spaceBefore=10, spaceAfter=4)
body = ParagraphStyle("body", parent=styles["Normal"], spaceAfter=4, leading=14)


def build_resume(filename: str, sections: list[tuple[str, str]]):
    path = os.path.join(OUT_DIR, filename)
    doc = SimpleDocTemplate(
        path, pagesize=letter,
        leftMargin=0.8 * inch, rightMargin=0.8 * inch,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
    )
    story = []
    for i, (heading, text) in enumerate(sections):
        style = h1 if i == 0 else h2
        story.append(Paragraph(heading, style))
        for line in text.strip().split("\n"):
            story.append(Paragraph(line.strip(), body))
        story.append(Spacer(1, 6))
    doc.build(story)
    print(f"Wrote {path}")


# --------------------------------------------------------------------------- #
# Resume A: strong Data Scientist match
# --------------------------------------------------------------------------- #
resume_a = [
    ("Priya Sharma", "Email: priya.sharma@example.com | Phone: (555) 111-2222"),
    ("Summary", """
    Data Scientist with 5 years of experience building and deploying machine
    learning models for e-commerce and fintech companies. Skilled in Python,
    deep learning, and cloud-based ML deployment.
    """),
    ("Skills", """
    Python, pandas, NumPy, scikit-learn, TensorFlow, PyTorch, SQL,
    A/B testing and experimental design, AWS (SageMaker, S3, EC2),
    Matplotlib, Seaborn, Tableau, MLflow, Airflow, Docker, Spark
    """),
    ("Experience", """
    Senior Data Scientist, FinEdge Analytics (2022-Present)
    - Built a churn prediction model using scikit-learn and TensorFlow,
      improving retention by 12%.
    - Deployed models to production on AWS SageMaker with MLflow tracking
      and Airflow-orchestrated retraining pipelines.
    - Designed and analyzed A/B tests for pricing experiments.

    Data Scientist, ShopWave Inc. (2020-2022)
    - Built recommendation systems using PyTorch, increasing conversion by 8%.
    - Used Spark for large-scale feature engineering on 500M+ row datasets.
    - Presented insights to executive stakeholders via Tableau dashboards.
    """),
    ("Education", """
    M.S. in Computer Science, University of Washington (2020)
    B.S. in Statistics, University of Michigan (2018)
    """),
]

# --------------------------------------------------------------------------- #
# Resume B: moderate match - strong SWE, lighter on ML/stats
# --------------------------------------------------------------------------- #
resume_b = [
    ("James Carter", "Email: james.carter@example.com | Phone: (555) 333-4444"),
    ("Summary", """
    Backend Software Engineer with 4 years of experience building scalable
    web services. Recently transitioning toward data-focused roles, with
    some exposure to Python-based data analysis.
    """),
    ("Skills", """
    Python, Java, SQL, PostgreSQL, REST APIs, Docker, Kubernetes,
    pandas (basic), Git, CI/CD, AWS (EC2, Lambda)
    """),
    ("Experience", """
    Software Engineer, CloudBridge Systems (2021-Present)
    - Built and maintained REST APIs serving 2M+ daily requests.
    - Wrote internal Python scripts using pandas for weekly reporting.
    - Deployed services on AWS using Docker and Kubernetes.

    Junior Software Engineer, DataPort LLC (2020-2021)
    - Maintained SQL-based ETL pipelines feeding internal dashboards.
    - Collaborated with the analytics team on ad-hoc data pulls.
    """),
    ("Education", """
    B.S. in Computer Science, Ohio State University (2020)
    """),
]

# --------------------------------------------------------------------------- #
# Resume C: weak match - entry-level analyst, many missing skills
# --------------------------------------------------------------------------- #
resume_c = [
    ("Maria Lopez", "Email: maria.lopez@example.com | Phone: (555) 777-8888"),
    ("Summary", """
    Recent graduate with a background in business analytics and Excel-based
    reporting. Eager to grow into a data-focused role.
    """),
    ("Skills", """
    Excel, PowerPoint, basic SQL, Google Sheets, PowerBI (beginner),
    strong communication and presentation skills
    """),
    ("Experience", """
    Business Analyst Intern, RetailNow Corp. (Summer 2025)
    - Built weekly sales reports in Excel and PowerBI for the merchandising team.
    - Wrote basic SQL queries to pull data from the company's data warehouse.
    - Assisted with slide decks summarizing quarterly performance for leadership.
    """),
    ("Education", """
    B.A. in Business Administration, Concentration in Analytics,
    Arizona State University (2025)
    """),
]

if __name__ == "__main__":
    build_resume("Resume_A.pdf", resume_a)
    build_resume("Resume_B.pdf", resume_b)
    build_resume("Resume_C.pdf", resume_c)
