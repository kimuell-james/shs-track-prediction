def generate_dashboard_descriptions(predicted_data, actual_data, gender_data, age_data):
    desc = {}

    # 🟦 Predicted Track Distribution
    if predicted_data:
        top_pred = max(predicted_data, key=lambda d: d["count"])
        desc["predicted"] = (
            f"Most students are predicted to take the **{top_pred['predicted_track']}** track "
            f"({top_pred['count']} students), indicating the model favors this track based on academic trends."
        )
    else:
        desc["predicted"] = "No predicted track data available."

    # 🟩 Actual Track Distribution
    if actual_data:
        top_actual = max(actual_data, key=lambda d: d["count"])
        desc["actual"] = (
            f"The **{top_actual['actual_track']}** track has the highest actual enrollment "
            f"({top_actual['count']} students), reflecting current SHS preferences."
        )
    else:
        desc["actual"] = "No actual track data available."

    # 🧑 Gender Distribution
    if gender_data:
        total_males = sum(d["count"] for d in gender_data if d["gender"] == "Male")
        total_females = sum(d["count"] for d in gender_data if d["gender"] == "Female")
        dominant = "Male" if total_males > total_females else "Female"
        desc["gender"] = (
            f"The gender distribution shows a higher number of **{dominant.lower()}** students overall. "
            "Gender balance across tracks helps understand diversity in student preferences."
        )
    else:
        desc["gender"] = "No gender data available."

    # 🎂 Age Distribution
    if age_data:
        avg_age = round(sum(d["age"] * d["count"] for d in age_data) / sum(d["count"] for d in age_data), 2)
        desc["age"] = f"The average student age is **{avg_age} years old**, with most students between 14–16."
    else:
        desc["age"] = "No age distribution data available."

    return desc
