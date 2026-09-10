"""
Generates the final PDF report for the Tourism Experience Analytics project.
Run this once locally: python generate_report.py
Produces: Tourism_Experience_Analytics_Report.pdf

Fill in / adjust the CONTENT section at the top with your own numbers if they
differ from what's below (these were pulled from your actual script outputs).
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                 PageBreak, HRFlowable)
from reportlab.lib.enums import TA_CENTER

OUT_FILE = "Tourism_Experience_Analytics_Report.pdf"

# ---------------- Styles ----------------
styles = getSampleStyleSheet()
title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=24, spaceAfter=6, textColor=colors.HexColor("#0F766E"))
subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=13, textColor=colors.HexColor("#475569"), alignment=TA_CENTER, spaceAfter=4)
h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, textColor=colors.HexColor("#0F766E"), spaceBefore=18, spaceAfter=8)
h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12.5, textColor=colors.HexColor("#134E4A"), spaceBefore=10, spaceAfter=6)
body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.3, leading=15, spaceAfter=8)
bullet = ParagraphStyle("Bullet", parent=body, leftIndent=14, bulletIndent=4, spaceAfter=5)
caption = ParagraphStyle("Caption", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#64748B"), alignment=TA_CENTER, spaceAfter=14)

TABLE_HEADER_BG = colors.HexColor("#0F766E")
TABLE_ALT_BG = colors.HexColor("#F0FDFA")

def make_table(data, col_widths=None):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.3),
        ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]
    for row in range(1, len(data), 2):
        style.append(("BACKGROUND", (0, row), (-1, row), TABLE_ALT_BG))
    t.setStyle(TableStyle(style))
    return t

def placeholder_box(label, height=2.3 * inch):
    """A dashed-border box for the user to paste their own chart/screenshot into."""
    from reportlab.platypus import Table as PBTable
    t = PBTable([[Paragraph(f"[ Insert here: {label} ]", ParagraphStyle(
        "ph", parent=body, alignment=TA_CENTER, textColor=colors.HexColor("#94A3B8")))]],
        colWidths=[6.3 * inch], rowHeights=[height])
    t.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#94A3B8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    return t

story = []

# =========================================================
# TITLE PAGE
# =========================================================
story.append(Spacer(1, 1.6 * inch))
story.append(Paragraph("Tourism Experience Analytics", title_style))
story.append(Paragraph("Classification, Prediction & Recommendation System", subtitle_style))
story.append(Spacer(1, 0.3 * inch))
story.append(HRFlowable(width="60%", thickness=1.2, color=colors.HexColor("#0F766E"), hAlign="CENTER"))
story.append(Spacer(1, 0.4 * inch))
story.append(Paragraph(
    "A data-driven system for predicting attraction ratings, classifying visit modes, "
    "and generating personalized attraction recommendations for tourism platforms.",
    ParagraphStyle("sub2", parent=body, alignment=TA_CENTER, fontSize=11, textColor=colors.HexColor("#475569"))
))
story.append(Spacer(1, 1.5 * inch))
story.append(Paragraph("Prepared by: _______________________", ParagraphStyle("prep", parent=body, alignment=TA_CENTER)))
story.append(Paragraph("Date: _______________________", ParagraphStyle("date", parent=body, alignment=TA_CENTER)))
story.append(PageBreak())

# =========================================================
# EXECUTIVE SUMMARY
# =========================================================
story.append(Paragraph("Executive Summary", h1))
story.append(Paragraph(
    "This project analyzes tourism transaction data covering 52,930 visits by 33,530 users "
    "across 9 attraction types and multiple regions worldwide. Three machine learning "
    "components were built: a regression model to predict attraction ratings, a classification "
    "model to predict visit mode (Business, Family, Couples, Friends, etc.), and a hybrid "
    "recommendation system combining collaborative and content-based filtering. All three are "
    "deployed in an interactive Streamlit application allowing users to input their profile and "
    "receive predictions and personalized attraction recommendations.", body
))
story.append(Paragraph(
    "Key finding: attraction ratings and visit mode are only weakly explained by geography and "
    "timing alone (regression R² ≈ 0.10, classification accuracy ≈ 49%), indicating that "
    "individual user experience and preference — not demographics — are the dominant drivers of "
    "satisfaction. This is itself an actionable insight: personalization strategies should "
    "weight individual history over broad demographic segmentation.", body
))

# =========================================================
# 1. DATA PREPARATION
# =========================================================
story.append(Paragraph("1. Data Preparation", h1))
story.append(Paragraph(
    "Nine source tables were combined: Transaction, User, City, Country, Region, Continent, "
    "Attraction Type, Visit Mode, and Item (attraction) data.", body
))
story.append(Paragraph("Cleaning steps performed", h2))
cleaning_steps = [
    "Stripped and standardized text fields (trimmed whitespace, title-cased categorical values) across all 9 tables.",
    "Dropped transaction rows with missing or invalid Rating values (kept only ratings in the valid 1-5 range).",
    "Dropped transaction rows with missing VisitMode, since it is a required classification target.",
    "Removed exact duplicate transaction records.",
    "Standardized VisitYear and VisitMonth as consistent integer fields for time-based analysis.",
    "Left-joined User → City → Country → Region → Continent, and Item (attraction) → Attraction Type, "
    "and Transaction → Visit Mode lookup, into a single master dataset.",
    "Filled remaining categorical gaps (8 rows missing city/country information after merge) with 'Unknown' "
    "rather than dropping — a negligible fraction of the 52,930-row dataset.",
]
for s in cleaning_steps:
    story.append(Paragraph(f"• {s}", bullet))

story.append(Paragraph("Result", h2))
data_summary = [
    ["Metric", "Value"],
    ["Raw transaction rows", "52,930"],
    ["Rows after cleaning", "52,930"],
    ["Unique users", "33,530"],
    ["Final merged dataset", "52,930 rows × 23 columns"],
    ["Missing values after merge", "8 rows (city/country), filled as 'Unknown'"],
]
story.append(make_table(data_summary, col_widths=[3 * inch, 3.3 * inch]))

# =========================================================
# 2. EXPLORATORY DATA ANALYSIS
# =========================================================
story.append(PageBreak())
story.append(Paragraph("2. Exploratory Data Analysis", h1))
story.append(Paragraph(
    "Eight visualizations were produced covering user distribution, attraction popularity, "
    "rating patterns, and seasonality. Key numeric findings from the attraction-type breakdown:", body
))

eda_table = [
    ["Attraction Type", "Avg Rating", "Visit Count"],
    ["Water Parks", "4.65", "6,429"],
    ["Caverns & Caves", "4.50", "135"],
    ["Nature & Wildlife Areas", "4.27", "13,251"],
    ["National Parks", "4.42", "511"],
    ["History Museums", "4.26", "978"],
    ["Religious Sites", "4.21", "6,711"],
    ["Points of Interest & Landmarks", "4.13", "6,252"],
    ["Beaches", "3.85", "10,917"],
    ["Historic Sites", "3.54", "798"],
]
story.append(make_table(eda_table, col_widths=[3 * inch, 1.4 * inch, 1.4 * inch]))
story.append(Spacer(1, 10))

story.append(Paragraph("Chart 1 — Visits by Continent", h2))
story.append(placeholder_box("reports/figures/01_visits_by_continent.png"))
story.append(Paragraph("Insight: [add 1 sentence on which continent dominates visit volume and why that might be.]", caption))

story.append(Paragraph("Chart 2 — Top 10 Countries by Visits", h2))
story.append(placeholder_box("reports/figures/02_top_countries.png"))
story.append(Paragraph("Insight: [add 1 sentence on geographic concentration of users.]", caption))

story.append(PageBreak())
story.append(Paragraph("Chart 3 — Average Rating by Attraction Type", h2))
story.append(placeholder_box("reports/figures/03_avgrating_by_type.png"))
story.append(Paragraph(
    "Insight: Water Parks earn the highest average rating (4.65) despite moderate traffic, while "
    "Beaches — the single most-visited category (10,917 visits) — score lowest among high-traffic "
    "types (3.85). This gap suggests a service-quality opportunity: beaches may be underdelivering "
    "relative to visitor expectations.", caption
))

story.append(Paragraph("Chart 4 — Top 10 Most-Visited Attractions", h2))
story.append(placeholder_box("reports/figures/04_top_attractions.png"))
story.append(Paragraph("Insight: [add 1 sentence naming the top attraction and any pattern across the top 10.]", caption))

story.append(Paragraph("Chart 5 — Visit Mode vs Continent", h2))
story.append(placeholder_box("reports/figures/05_visitmode_vs_continent.png"))
story.append(Paragraph("Insight: [add 1 sentence on whether certain continents skew toward Business/Family/Couples travel.]", caption))

story.append(PageBreak())
story.append(Paragraph("Chart 6 — Seasonality of Visits", h2))
story.append(placeholder_box("reports/figures/06_seasonality.png"))
story.append(Paragraph("Insight: [add 1 sentence on peak visiting months.]", caption))

story.append(Paragraph("Chart 7 — Rating Distribution", h2))
story.append(placeholder_box("reports/figures/07_rating_distribution.png"))
story.append(Paragraph("Insight: [add 1 sentence on the overall shape — e.g. skewed toward high ratings.]", caption))

story.append(Paragraph("Chart 8 — Correlation Matrix (numeric features)", h2))
story.append(placeholder_box("reports/figures/08_correlation_matrix.png"))
story.append(Paragraph("Insight: [add 1 sentence on any notable correlation, or note the general weakness of linear relationships — consistent with the modest regression R² below.]", caption))

# =========================================================
# 3. MODEL PERFORMANCE
# =========================================================
story.append(PageBreak())
story.append(Paragraph("3. Model Performance", h1))

story.append(Paragraph("3.1 Regression — Predicting Rating", h2))
story.append(Paragraph(
    "Features used: Continent, Region, Country, City, Attraction Type, Visit Year, Visit Month.", body
))
reg_table = [
    ["Model", "R²", "MAE", "RMSE"],
    ["Linear Regression", "0.029", "0.749", "0.956"],
    ["Random Forest", "-0.015", "0.748", "0.978"],
    ["XGBoost (best)", "0.103", "0.718", "0.919"],
]
story.append(make_table(reg_table, col_widths=[2.3 * inch, 1.3 * inch, 1.3 * inch, 1.3 * inch]))
story.append(Paragraph(
    "XGBoost performed best but explains only ~10% of rating variance. This indicates ratings are "
    "driven substantially by factors not captured by geography or visit timing alone — most likely "
    "individual subjective experience, service quality on the day, or attraction-specific factors "
    "not present in this feature set. Adding user-level historical behavior (e.g. a user's average "
    "past rating) would likely improve this significantly in a future iteration.", body
))

story.append(Paragraph("3.2 Classification — Predicting Visit Mode", h2))
clf_table = [
    ["Model", "Accuracy", "Precision", "Recall", "F1"],
    ["Logistic Regression", "0.282", "0.400", "0.282", "0.316"],
    ["Random Forest (best F1)", "0.440", "0.463", "0.440", "0.448"],
    ["XGBoost", "0.492", "0.483", "0.492", "0.441"],
]
story.append(make_table(clf_table, col_widths=[2.3 * inch, 1.15 * inch, 1.15 * inch, 1.0 * inch, 0.9 * inch]))
story.append(Paragraph(
    "Random Forest was selected as the best model by weighted F1-score, balancing performance across "
    "the imbalanced visit-mode classes better than Logistic Regression. The minority class (Class 1, "
    "n=125) is the hardest to predict, as expected given limited training examples. As with the "
    "regression task, accuracy could likely be improved by adding user-level features (e.g. a user's "
    "historical visit mode pattern).", body
))

# =========================================================
# 4. RECOMMENDATION SYSTEM
# =========================================================
story.append(PageBreak())
story.append(Paragraph("4. Recommendation System", h1))
story.append(Paragraph(
    "Two complementary approaches were implemented on a 33,530 × 30 user-item ratings matrix:", body
))
story.append(Paragraph("• <b>Collaborative filtering</b> — matrix factorization (SVD) on the user-item rating matrix, "
                        "recommending attractions similar users rated highly.", bullet))
story.append(Paragraph("• <b>Content-based filtering</b> — cosine similarity over attraction features (type, location), "
                        "recommending attractions similar to ones a user already liked.", bullet))

story.append(Paragraph("Sample output", h2))
sample_table = [
    ["Method", "Input", "Top Recommendations"],
    ["Collaborative Filtering", "User 14",
     "Coban Rondo Waterfall, Goa Cina Beach, Sempu Island, Kalibiru National Park, Balekambang Beach"],
    ["Content-Based Filtering", "Sacred Monkey Forest Sanctuary",
     "Sempu Island, Seminyak Beach, Nusa Dua Beach, Sanur Beach, Waterbom Bali"],
]
story.append(make_table(sample_table, col_widths=[1.7 * inch, 1.7 * inch, 3 * inch]))
story.append(Paragraph(
    "The content-based example demonstrates strong geographic and thematic coherence (all "
    "recommendations for the Bali-based input are also Bali attractions), validating the "
    "similarity approach for guiding users toward relevant nearby experiences.", body
))

# =========================================================
# 5. BUSINESS INSIGHTS
# =========================================================
story.append(Paragraph("5. Business Insights & Recommendations", h1))
insights = [
    "Beaches drive the highest visit volume but underperform on satisfaction (3.85 avg rating) — "
    "flag for a service-quality review (crowding, cleanliness, amenities) before further marketing spend.",
    "Water Parks and Caverns/Caves post the highest satisfaction scores despite lower traffic — "
    "strong candidates for increased promotion, as they are more likely to convert visits into "
    "positive reviews and repeat visits.",
    "Nature & Wildlife Areas are both the highest-volume category and solidly rated (4.27) — "
    "the platform's core reliable offering; worth protecting and expanding capacity here.",
    "Weak regression/classification performance from demographic features alone suggests marketing "
    "and recommendation strategy should lean on individual user history (past ratings, past visit "
    "modes) rather than broad geographic segmentation for personalization.",
    "The content-based recommender's strong geographic clustering (e.g. the Bali example) supports "
    "using it for 'more like this' in-app suggestions immediately, even while the classification "
    "model is further improved.",
]
for i in insights:
    story.append(Paragraph(f"• {i}", bullet))

# =========================================================
# 6. STREAMLIT APPLICATION
# =========================================================
story.append(PageBreak())
story.append(Paragraph("6. Streamlit Application", h1))
story.append(Paragraph(
    "An interactive Streamlit application was built with four tabs: Overview (dataset metrics and "
    "charts), Rating Prediction, Visit Mode Prediction, and Recommendations (both collaborative and "
    "content-based). Location inputs use cascading dropdowns (Continent → Region → Country → City) "
    "so users only see valid combinations at each step.", body
))

story.append(Paragraph("Screenshot — Overview tab", h2))
story.append(placeholder_box("Screenshot of the Overview tab"))

story.append(Paragraph("Screenshot — Rating Prediction result", h2))
story.append(placeholder_box("Screenshot of a rating prediction result"))

story.append(Paragraph("Screenshot — Recommendations tab", h2))
story.append(placeholder_box("Screenshot of a recommendation result"))

# =========================================================
# 7. CONCLUSION
# =========================================================
story.append(PageBreak())
story.append(Paragraph("7. Conclusion", h1))
story.append(Paragraph(
    "This project delivers a complete, end-to-end tourism analytics pipeline: cleaned and merged "
    "data across nine source tables, exploratory analysis surfacing concrete service-quality and "
    "demand insights, two supervised models (regression and classification) with transparent "
    "performance evaluation, a working hybrid recommendation system, and a deployed interactive "
    "application. The most valuable finding may be a negative one — that geography and timing alone "
    "are weak predictors of satisfaction and behavior — which itself directs future work toward "
    "richer, user-history-based features.", body
))

doc = SimpleDocTemplate(OUT_FILE, pagesize=letter,
                         topMargin=0.8 * inch, bottomMargin=0.8 * inch,
                         leftMargin=0.85 * inch, rightMargin=0.85 * inch)
doc.build(story)
print(f"Report generated: {OUT_FILE}")
