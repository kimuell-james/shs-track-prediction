# report_generator.py
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
)
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from predictor.evaluate_model import evaluate_active_model
from django.db.models import Count
from predictor.models import SchoolYear,Student, StudentGrade

def plot_to_base64(fig):
    """Convert Matplotlib figure to base64 string."""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()

def generate_pdf_report(filename, sy_id):
    # --- Retrieve school year ---
    school_year = SchoolYear.objects.get(sy_id=sy_id)

    # Run evaluation
    results = evaluate_active_model(school_year)

    long_size = (8.5 * inch, 13 * inch)

    doc = SimpleDocTemplate(
        filename,
        pagesize=long_size,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    styles = getSampleStyleSheet()
    elements = []

    if results.get("no_data"):
        elements.append(Paragraph("No data available for this school year.", styles["Normal"]))
        doc.build(elements)
        return

    # Extract evaluation results
    accuracy = results["accuracy"]
    roc_auc = results["roc_auc"]
    report = results["report"]
    conf_matrix = results["conf_matrix"]
    cm_base64 = results["cm_base64"]
    roc_base64 = results["roc_base64"]
    analysis = results["analysis"]
    df = results["students_df"]

    # --- Title ---
    elements.append(Paragraph("<b>SHS Track Insight Report</b>", styles["Title"]))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(f"School Year: <b>{school_year.school_year}</b>", styles["Heading2"]))
    elements.append(Spacer(1, 12))

    # --- Overview Stats ---
    total_students = Student.objects.filter(sy=school_year).count()
    predicted_count = Student.objects.filter(sy=school_year, predicted_track__isnull=False).count()
    actual_count = Student.objects.filter(sy=school_year, actual_track__isnull=False).count()

    overview_data = [
        ["Metric", "Count"],
        ["Total Students", total_students],
        ["With Predicted Track", predicted_count],
        ["With Actual Track", actual_count],
    ]
    overview_table = Table(overview_data, hAlign='LEFT')
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT')
    ]))

    overview_section = [
        Paragraph("<b>Data Overview</b>", styles["Heading2"]),
        overview_table,
        Spacer(1, 0.3 * inch),
    ]
    elements.append(KeepTogether(overview_section))

    # --- Model Performance Metrics ---
    model_data = [
        ["Metric", "Value"],
        ["Accuracy", f"{accuracy*100:.2f}%"],
        ["ROC-AUC", f"{roc_auc:.3f}"],
        ["Precision (Academic)", f"{report['Academic']['precision']:.2f}"],
        ["Recall (Academic)", f"{report['Academic']['recall']:.2f}"],
        ["Precision (TVL)", f"{report['TVL']['precision']:.2f}"],
        ["Recall (TVL)", f"{report['TVL']['recall']:.2f}"],
    ]
    model_table = Table(model_data, hAlign='LEFT')
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ]))

    model_section = [
        Paragraph("<b>Model Evaluation Metrics</b>", styles["Heading2"]),
        model_table,
        Spacer(1, 0.3 * inch),
    ]
    elements.append(KeepTogether(model_section))

    # --- Average Grades per Track ---
    data = list(
        StudentGrade.objects.filter(student_id__sy=school_year)
        .values("student_id__actual_track",
                *[f"g{i}_{subj}" for i in range(7, 11) for subj in [
                    "filipino", "english", "math", "science", "ap", "tle", "mapeh", "esp"
                ]])
    )

    df = pd.DataFrame(data)
    if not df.empty:
        df.rename(columns={"student_id__actual_track": "track"}, inplace=True)
        grade_cols = [col for col in df.columns if col.startswith("g")]
        long_df = df.melt(id_vars=["track"], value_vars=grade_cols,
                        var_name="subject", value_name="grade")
        long_df["grade_level"] = long_df["subject"].str.extract(r"(g\d+)")
        long_df["subject_name"] = long_df["subject"].str.replace(r"g\d+_", "", regex=True)
        avg_df = (long_df.groupby(["grade_level", "subject_name", "track"])["grade"]
                .mean().unstack(fill_value=0).round(2).reset_index())

        elements.append(Paragraph("<b>Average Grades per Track (per Grade Level)</b>", styles["Heading2"]))
        grade_order = sorted(avg_df["grade_level"].unique(), key=lambda x: int(x[1:]))

        for grade in grade_order:
            sub_df = avg_df[avg_df["grade_level"] == grade]
            sub_table = [list(sub_df.columns)] + sub_df.values.tolist()

            t = Table(sub_table, hAlign='LEFT')
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
            ]))

            # group heading + table + spacer together
            elements.append(
                KeepTogether([
                    Paragraph(f"<b>{grade.upper()}</b>", styles["Heading3"]),
                    t,
                    Spacer(1, 0.2 * inch),
                ])
            )
    else:
        elements.append(Paragraph("No grade data available.", styles["Normal"]))

    # --- Distributions ---
    predicted_distribution = Student.objects.filter(sy=school_year).values("predicted_track").annotate(count=Count("predicted_track"))
    actual_distribution = Student.objects.filter(sy=school_year).values("actual_track").annotate(count=Count("actual_track"))
    gender_distribution = Student.objects.filter(sy=school_year).values("actual_track", "gender").annotate(count=Count("gender"))
    age_distribution = Student.objects.filter(sy=school_year).values("actual_track", "age").annotate(count=Count("age")).order_by("age")

    track_colors = {
        "Academic": "#36A2EB",
        "TVL": "#379145"
    }

    # Convert distribution data into tables
    def make_table(data, columns):
        """Return a styled ReportLab Table object from given data."""
        if not data:
            return Paragraph("No data available.", styles["Normal"])
        table_data = [columns] + [[str(row.get(col, "")) for col in columns] for row in data]
        t = Table(table_data, hAlign="LEFT")
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        return t

    # --- Predicted Track Distribution (Pie) ---
    predicted_df = pd.DataFrame(list(predicted_distribution))
    if not predicted_df.empty:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(
            predicted_df["count"], 
            labels=predicted_df["predicted_track"].fillna("Unknown"),
            autopct="%1.1f%%",
            startangle=90,
            colors=[track_colors.get(t, "#CCCCCC") for t in predicted_df["predicted_track"]]
        )
        # ax.set_title("Predicted Track Distribution", fontsize=14, weight="bold", pad=15)
        img = Image(BytesIO(base64.b64decode(plot_to_base64(fig))))
        img._restrictSize(300, 300)
        
        # Use make_table() to build the table object
        table_obj = make_table(list(predicted_distribution), ["predicted_track", "count"])

        # Combine chart and table side by side
        combined = Table([[img, table_obj]], colWidths=[3.2 * inch, 3.2 * inch])
        combined.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))

        section = [
            Paragraph("<b>Predicted Track Distribution</b>", styles["Heading2"]),
            combined,
            Spacer(1, 0.3 * inch),
        ]
        elements.append(KeepTogether(section))

    # --- Actual Track Distribution (Pie) ---
    actual_df = pd.DataFrame(list(actual_distribution))
    if not actual_df.empty:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(
            actual_df["count"], 
            labels=actual_df["actual_track"].fillna("Unknown"),
            autopct="%1.1f%%",
            startangle=90,
            colors=[track_colors.get(t, "#CCCCCC") for t in actual_df["actual_track"]]
        )
        # ax.set_title("Actual Track Distribution", fontsize=14, weight="bold", pad=15)
        img = Image(BytesIO(base64.b64decode(plot_to_base64(fig))))
        img._restrictSize(300, 300)
        
        # Use make_table() to build the table object
        table_obj = make_table(list(actual_distribution), ["actual_track", "count"])

        # Combine chart and table side by side
        combined = Table([[img, table_obj]], colWidths=[3.2 * inch, 3.2 * inch])
        combined.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))

        section = [
            Paragraph("<b>Actual Track Distribution</b>", styles["Heading2"]),
            combined,
            Spacer(1, 0.3 * inch),
        ]
        elements.append(KeepTogether(section))

    # --- Age Distribution per Track (Stacked Bar) ---
    age_df = pd.DataFrame(list(age_distribution))
    if not age_df.empty:
        pivot_age = age_df.pivot(index="age", columns="actual_track", values="count").fillna(0)
        pivot_age.plot(kind="bar", stacked=True, figsize=(6, 4),
                       color=["#36A2EB", "#379145", "#FFCE56", "#4BC0C0", "#9966FF", "#FF9F40"])
        # plt.title("Age Distribution per Track", fontsize=14, weight="bold", pad=10)
        plt.ylabel("Count")
        plt.xlabel("Age")
        plt.tight_layout()
        img = Image(BytesIO(base64.b64decode(plot_to_base64(plt.gcf()))))
        img._restrictSize(450, 350)

        table_obj = make_table(list(age_distribution), ["actual_track", "age", "count"])

        section = [
            Paragraph("<b>Age Distribution per Track</b>", styles["Heading2"]),
            img,
            Spacer(1, 0.2 * inch),
            table_obj,
            Spacer(1, 0.3 * inch),
        ]
        elements.append(KeepTogether(section))

    # --- Gender Distribution per Track (Bar) ---
    gender_df = pd.DataFrame(list(gender_distribution))
    if not gender_df.empty:
        pivot_gender = gender_df.pivot(index="actual_track", columns="gender", values="count").fillna(0)
        pivot_gender.plot(kind="bar", figsize=(5, 4), color=["#FF6384", "#36A2EB"])
        # plt.title("Gender Distribution per Track", fontsize=14, weight="bold", pad=10)
        plt.ylabel("Count")
        plt.xlabel("Track")
        plt.xticks(rotation=0)
        plt.tight_layout()
        img = Image(BytesIO(base64.b64decode(plot_to_base64(plt.gcf()))))
        img._restrictSize(400, 300)

        table_obj = make_table(list(gender_distribution), ["actual_track", "gender", "count"])

        section = [
            Paragraph("<b>Gender Distribution per Track</b>", styles["Heading2"]),
            img,
            Spacer(1, 0.2 * inch),
            table_obj,
            Spacer(1, 0.3 * inch),
        ]
        elements.append(KeepTogether(section))

    # --- Confusion Matrix ---
    cm_image_data = base64.b64decode(cm_base64)
    cm_image = Image(BytesIO(cm_image_data))
    cm_image._restrictSize(400, 300)

    conf_matrix_section = [
            Paragraph("<b>Confusion Matrix</b>", styles["Heading2"]),
            cm_image,
            Spacer(1, 10),
        ]
    elements.append(KeepTogether(conf_matrix_section))
    

    # --- Embedded ROC Curve Image ---
    roc_image_data = base64.b64decode(roc_base64)
    roc_image = Image(BytesIO(roc_image_data))
    roc_image._restrictSize(400, 300)

    roc_section = [
            Paragraph("<b>ROC Curve</b>", styles["Heading2"]),
            roc_image,
            Spacer(1, 12),
        ]
    elements.append(KeepTogether(roc_section))

    # --- Analytical Insights ---
    elements.append(Paragraph("<b>Model Analysis</b>", styles["Heading2"]))

    for section, text in analysis.items():
        section_block = [
            Paragraph(f"<b>{section}</b>", styles["Heading3"]) if section else None,
            Paragraph(text, styles["Normal"]),
            Spacer(1, 6)
        ]
        # Filter out None values (in case section is empty)
        section_block = [item for item in section_block if item]

        elements.append(KeepTogether(section_block))

    

    # --- Footer ---
    elements.append(Paragraph("<b>End of Report</b>", styles["Italic"]))

    # Build PDF
    doc.build(elements)
