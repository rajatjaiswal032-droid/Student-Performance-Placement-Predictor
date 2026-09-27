import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance & Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PERFORMANCE_DATA = BASE_DIR / "data" / "raw" / "student_performance.csv"
PERFORMANCE_TARGETS = BASE_DIR / "data" / "raw" / "student_performance_targets.csv"
PLACEMENT_DATA = BASE_DIR / "data" / "raw" / "placement_data.csv"

PERFORMANCE_MODEL = BASE_DIR / "models" / "performance_model.joblib"
PERFORMANCE_PREPROCESSOR = BASE_DIR / "models" / "performance_preprocessor.joblib"

PLACEMENT_MODEL = BASE_DIR / "models" / "placement_model.joblib"
PLACEMENT_PREPROCESSOR = BASE_DIR / "models" / "placement_preprocessor.joblib"


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_performance_data():
    features = pd.read_csv(PERFORMANCE_DATA)
    targets = pd.read_csv(PERFORMANCE_TARGETS)

    df = features.copy()

    for column in targets.columns:
        df[column] = targets[column]

    return df


@st.cache_data
def load_placement_data():
    return pd.read_csv(PLACEMENT_DATA)


@st.cache_resource
def load_models():

    performance_model = joblib.load(PERFORMANCE_MODEL)
    performance_preprocessor = joblib.load(
        PERFORMANCE_PREPROCESSOR
    )

    placement_model = joblib.load(PLACEMENT_MODEL)
    placement_preprocessor = joblib.load(
        PLACEMENT_PREPROCESSOR
    )

    return (
        performance_model,
        performance_preprocessor,
        placement_model,
        placement_preprocessor
    )


performance_df = load_performance_data()
placement_df = load_placement_data()

(
    performance_model,
    performance_preprocessor,
    placement_model,
    placement_preprocessor
) = load_models()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎓 Student Analytics")

st.sidebar.markdown(
    """
    ### Student Performance & Placement Predictor

    Machine-learning dashboard for:

    - 📊 Academic analysis
    - 🎯 Performance prediction
    - 💼 Placement prediction
    - 🤖 Model evaluation
    - 💡 Student insights
    """
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Dataset & EDA",
        "Performance Analysis",
        "Performance Prediction",
        "Placement Prediction",
        "Model Evaluation",
        "Insights",
        "About"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("BTech Data Science Engineering Project")


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.title("🎓 Student Performance & Placement Predictor")

    st.markdown(
        """
        ### Data-Driven Student Analytics

        An end-to-end Data Science and Machine Learning project
        for analyzing academic performance and placement outcomes.
        """
    )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Academic Records",
            f"{len(performance_df):,}"
        )

    with col2:
        st.metric(
            "Placement Records",
            f"{len(placement_df):,}"
        )

    with col3:
        st.metric(
            "Academic Features",
            "30"
        )

    with col4:
        st.metric(
            "Placement Features",
            "8"
        )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📚 Performance Prediction")

        st.write(
            """
            Predict a student's final academic score using
            demographic, family, academic-behavior and
            attendance-related features.
            """
        )

        st.info("Target: G3 / Final Academic Score")

    with col2:

        st.subheader("💼 Placement Prediction")

        st.write(
            """
            Estimate placement outcomes using academic,
            communication, project and skill-related features.
            """
        )

        st.info("Target: Placement")

    st.markdown("---")

    st.info(
        "Model predictions are statistical estimates and should "
        "not be treated as guarantees."
    )


# ============================================================
# DATASET & EDA
# ============================================================

elif page == "Dataset & EDA":

    st.title("📊 Dataset & Exploratory Data Analysis")

    tab1, tab2 = st.tabs(
        ["Academic Dataset", "Placement Dataset"]
    )

    # --------------------------------------------------------
    # ACADEMIC DATASET
    # --------------------------------------------------------

    with tab1:

        st.subheader("Academic Dataset")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Students",
                f"{len(performance_df):,}"
            )

        with col2:
            st.metric(
                "Features",
                "30"
            )

        with col3:
            st.metric(
                "Target",
                "G3"
            )

        st.markdown("### Dataset Preview")

        st.dataframe(
            performance_df.head(10),
            use_container_width=True
        )

        st.markdown("### Final Score Distribution")

        score_counts = (
            performance_df["G3"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(score_counts)

        st.markdown("### Missing Values")

        missing_values = performance_df.isnull().sum()

        missing_table = pd.DataFrame({
            "Column": missing_values.index,
            "Missing Values": missing_values.values
        })

        st.dataframe(
            missing_table,
            use_container_width=True
        )

    # --------------------------------------------------------
    # PLACEMENT DATASET
    # --------------------------------------------------------

    with tab2:

        st.subheader("Placement Dataset")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Students",
                f"{len(placement_df):,}"
            )

        with col2:
            st.metric(
                "Features Used",
                "8"
            )

        with col3:
            st.metric(
                "Target",
                "Placement"
            )

        st.markdown("### Dataset Preview")

        st.dataframe(
            placement_df.head(10),
            use_container_width=True
        )

        st.markdown("### Placement Distribution")

        placement_counts = placement_df["Placement"].value_counts()

        st.bar_chart(placement_counts)

        st.markdown("### Placement Summary")

        st.dataframe(
            placement_counts.rename_axis("Placement")
            .reset_index(name="Students"),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PERFORMANCE ANALYSIS
# ============================================================

elif page == "Performance Analysis":

    st.title("📚 Performance Analysis")

    st.markdown(
        """
        Explore relationships between student characteristics
        and the final academic score (**G3**).
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Average Final Score",
            f"{performance_df['G3'].mean():.2f}"
        )

    with col2:
        st.metric(
            "Highest Score",
            f"{performance_df['G3'].max():.0f}"
        )

    with col3:
        st.metric(
            "Lowest Score",
            f"{performance_df['G3'].min():.0f}"
        )

    st.markdown("---")

    st.subheader("Study Time vs Final Score")

    studytime_analysis = (
        performance_df
        .groupby("studytime")["G3"]
        .mean()
        .round(2)
    )

    st.bar_chart(studytime_analysis)

    st.markdown("---")

    st.subheader("Failures vs Final Score")

    failures_analysis = (
        performance_df
        .groupby("failures")["G3"]
        .mean()
        .round(2)
    )

    st.bar_chart(failures_analysis)

    st.markdown("---")

    st.subheader("Absences vs Final Score")

    absence_analysis = (
        performance_df
        .groupby("absences")["G3"]
        .mean()
        .sort_index()
        .round(2)
    )

    st.line_chart(absence_analysis)

    st.info(
        "These relationships describe associations in the dataset "
        "and should not be interpreted as proof of causation."
    )


# ============================================================
# PERFORMANCE PREDICTION
# ============================================================

elif page == "Performance Prediction":

    st.title("🎯 Academic Performance Prediction")

    st.markdown(
        """
        Enter student information to estimate the **Final Academic
        Score (G3)** using the trained Random Forest Regressor.
        """
    )

    st.markdown("---")

    st.subheader("👤 Student Information")

    col1, col2 = st.columns(2)

    with col1:

        school = st.selectbox(
            "School",
            ["GP", "MS"]
        )

        sex = st.selectbox(
            "Sex",
            ["M", "F"]
        )

        age = st.number_input(
            "Age",
            min_value=15,
            max_value=25,
            value=17
        )

        address = st.selectbox(
            "Address",
            ["U", "R"]
        )

        famsize = st.selectbox(
            "Family Size",
            ["GT3", "LE3"]
        )

        pstatus = st.selectbox(
            "Parent Cohabitation Status",
            ["T", "A"]
        )

        medu = st.number_input(
            "Mother's Education",
            min_value=0,
            max_value=4,
            value=2
        )

        fedu = st.number_input(
            "Father's Education",
            min_value=0,
            max_value=4,
            value=2
        )

        mjob = st.selectbox(
            "Mother's Job",
            ["teacher", "health", "services", "at_home", "other"]
        )

        fjob = st.selectbox(
            "Father's Job",
            ["teacher", "health", "services", "at_home", "other"]
        )

        reason = st.selectbox(
            "Reason for Choosing School",
            ["course", "home", "reputation", "other"]
        )

        guardian = st.selectbox(
            "Guardian",
            ["mother", "father", "other"]
        )

        traveltime = st.number_input(
            "Travel Time",
            min_value=1,
            max_value=4,
            value=1
        )

        studytime = st.number_input(
            "Study Time",
            min_value=1,
            max_value=4,
            value=2
        )

        failures = st.number_input(
            "Past Class Failures",
            min_value=0,
            max_value=4,
            value=0
        )

    with col2:

        schoolsup = st.selectbox(
            "Extra Educational Support",
            ["yes", "no"]
        )

        famsup = st.selectbox(
            "Family Educational Support",
            ["yes", "no"]
        )

        paid = st.selectbox(
            "Extra Paid Classes",
            ["yes", "no"]
        )

        activities = st.selectbox(
            "Extra-Curricular Activities",
            ["yes", "no"]
        )

        nursery = st.selectbox(
            "Attended Nursery School",
            ["yes", "no"]
        )

        higher = st.selectbox(
            "Wants Higher Education",
            ["yes", "no"]
        )

        internet = st.selectbox(
            "Internet Access",
            ["yes", "no"]
        )

        romantic = st.selectbox(
            "Romantic Relationship",
            ["yes", "no"]
        )

        famrel = st.number_input(
            "Family Relationship Quality",
            min_value=1,
            max_value=5,
            value=4
        )

        freetime = st.number_input(
            "Free Time",
            min_value=1,
            max_value=5,
            value=3
        )

        goout = st.number_input(
            "Going Out",
            min_value=1,
            max_value=5,
            value=3
        )

        dalc = st.number_input(
            "Workday Alcohol Consumption",
            min_value=1,
            max_value=5,
            value=1
        )

        walc = st.number_input(
            "Weekend Alcohol Consumption",
            min_value=1,
            max_value=5,
            value=1
        )

        health = st.number_input(
            "Health Status",
            min_value=1,
            max_value=5,
            value=4
        )

        absences = st.number_input(
            "Absences",
            min_value=0,
            max_value=100,
            value=5
        )

    st.markdown("---")

    predict_performance = st.button(
        "🎯 Predict Final Academic Score",
        use_container_width=True
    )

    if predict_performance:

        input_data = pd.DataFrame({
            "school": [school],
            "sex": [sex],
            "age": [age],
            "address": [address],
            "famsize": [famsize],
            "Pstatus": [pstatus],
            "Medu": [medu],
            "Fedu": [fedu],
            "Mjob": [mjob],
            "Fjob": [fjob],
            "reason": [reason],
            "guardian": [guardian],
            "traveltime": [traveltime],
            "studytime": [studytime],
            "failures": [failures],
            "schoolsup": [schoolsup],
            "famsup": [famsup],
            "paid": [paid],
            "activities": [activities],
            "nursery": [nursery],
            "higher": [higher],
            "internet": [internet],
            "romantic": [romantic],
            "famrel": [famrel],
            "freetime": [freetime],
            "goout": [goout],
            "Dalc": [dalc],
            "Walc": [walc],
            "health": [health],
            "absences": [absences]
        })

        processed_input = performance_preprocessor.transform(
            input_data
        )

        predicted_score = performance_model.predict(
            processed_input
        )[0]

        predicted_score = max(
            0,
            min(20, predicted_score)
        )

        st.markdown("---")

        st.success(
            f"### 📚 Predicted Final Academic Score: "
            f"{predicted_score:.2f} / 20"
        )

        score_col1, score_col2 = st.columns(2)

        with score_col1:
            st.metric(
                "Predicted Score",
                f"{predicted_score:.2f} / 20"
            )

        with score_col2:
            st.metric(
                "Model",
                "Random Forest Regressor"
            )

        st.caption(
            "The prediction is an estimate generated from patterns "
            "learned from the training dataset."
        )


# ============================================================
# PLACEMENT PREDICTION
# ============================================================

elif page == "Placement Prediction":

    st.title("💼 Placement Prediction")

    st.markdown(
        """
        Enter student information to estimate the placement outcome.
        """
    )

    st.markdown("---")

    st.subheader("Student Information")

    col1, col2 = st.columns(2)

    with col1:

        iq = st.number_input(
            "IQ",
            min_value=0,
            max_value=200,
            value=100
        )

        prev_sem_result = st.number_input(
            "Previous Semester Result",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

        cgpa = st.number_input(
            "CGPA",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.1
        )

        academic_performance = st.number_input(
            "Academic Performance",
            min_value=1,
            max_value=10,
            value=5
        )

    with col2:

        internship = st.selectbox(
            "Internship Experience",
            ["Yes", "No"]
        )

        extra_curricular = st.number_input(
            "Extra-Curricular Score",
            min_value=0,
            max_value=10,
            value=5
        )

        communication = st.number_input(
            "Communication Skills",
            min_value=0,
            max_value=10,
            value=5
        )

        projects = st.number_input(
            "Projects Completed",
            min_value=0,
            max_value=10,
            value=2
        )

    st.markdown("---")

    predict_placement = st.button(
        "🔮 Predict Placement",
        use_container_width=True
    )

    if predict_placement:

        input_data = pd.DataFrame({
            "IQ": [iq],
            "Prev_Sem_Result": [prev_sem_result],
            "CGPA": [cgpa],
            "Academic_Performance": [academic_performance],
            "Internship_Experience": [internship],
            "Extra_Curricular_Score": [extra_curricular],
            "Communication_Skills": [communication],
            "Projects_Completed": [projects]
        })

        processed_input = placement_preprocessor.transform(
            input_data
        )

        prediction = placement_model.predict(
            processed_input
        )[0]

        probability = placement_model.predict_proba(
            processed_input
        )[0][1]

        st.markdown("---")

        if prediction == 1:

            st.success(
                "### ✅ Placement Prediction: YES"
            )

        else:

            st.warning(
                "### ⚠️ Placement Prediction: NO"
            )

        st.metric(
            "Estimated Placement Probability",
            f"{probability * 100:.2f}%"
        )

        st.caption(
            "This probability is a model estimate based on the "
            "training dataset and is not a guarantee of placement."
        )


# ============================================================
# MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.title("🤖 Model Evaluation")

    st.markdown(
        """
        The project uses separate machine-learning models for
        academic performance regression and placement classification.
        """
    )

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    st.subheader("📚 Performance Prediction")

    performance_results = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Random Forest Regressor"
        ],
        "MAE": [
            2.1564,
            2.0452
        ],
        "RMSE": [
            2.8618,
            2.8166
        ],
        "R²": [
            0.1602,
            0.1865
        ]
    })

    st.dataframe(
        performance_results,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "MAE and RMSE measure prediction error. R² measures "
        "explained variation on the evaluated test set."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # PLACEMENT
    # --------------------------------------------------------

    st.subheader("💼 Placement Prediction")

    placement_results = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest Classifier"
        ],
        "Accuracy": [
            0.8985,
            0.9995
        ],
        "Precision": [
            0.7490,
            1.0000
        ],
        "Recall": [
            0.5843,
            0.9970
        ],
        "F1 Score": [
            0.6565,
            0.9985
        ],
        "ROC-AUC": [
            0.9428,
            1.0000
        ]
    })

    st.dataframe(
        placement_results,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader("🔍 Cross-Validation")

    st.metric(
        "Random Forest 5-Fold ROC-AUC",
        "1.0000"
    )

    st.warning(
        """
        The placement Random Forest produced unusually high
        performance on this dataset. This should be investigated
        for dataset-specific structure or target-generation
        relationships before treating the result as expected
        real-world performance.
        """
    )


# ============================================================
# INSIGHTS
# ============================================================

elif page == "Insights":

    st.title("💡 Student Insights")

    st.subheader("📚 Academic Performance")

    st.markdown(
        """
        - The average final academic score in the dataset is
          approximately **11.91 / 20**.
        - Study time, previous failures and absences can be explored
          as factors associated with final academic performance.
        - The Random Forest Regressor achieved an **R² of 0.1865**
          on the evaluated test set.
        """
    )

    st.subheader("💼 Placement")

    st.markdown(
        """
        - The placement dataset contains **10,000 student records**.
        - The dataset contains more students marked as not placed
          than placed.
        - The placement model uses academic and skill-related
          variables such as CGPA, previous semester result,
          communication skills and completed projects.
        - The Random Forest produced very high evaluation scores
          on this dataset and therefore requires careful
          dataset-specific interpretation.
        """
    )

    st.subheader("⚠️ Important")

    st.info(
        """
        Machine-learning predictions represent statistical estimates.
        They should support analysis and decision-making rather than
        replace human judgment.
        """
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.title("ℹ️ About the Project")

    st.markdown(
        """
        ## Student Performance & Placement Predictor

        An end-to-end Data Science and Machine Learning project
        that analyzes academic performance and placement-related
        outcomes.

        ### Technologies

        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        - Matplotlib
        - Seaborn
        - Joblib
        - Streamlit

        ### Machine Learning

        **Performance Prediction**

        - Linear Regression
        - Random Forest Regressor

        **Placement Prediction**

        - Logistic Regression
        - Random Forest Classifier

        ### Workflow

        Data Collection → Data Cleaning → EDA →
        Feature Engineering → Model Training →
        Evaluation → Prediction → Dashboard

        ### Project Goal

        Demonstrate a complete Data Science workflow from
        raw datasets to machine-learning models and an
        interactive deployment-ready dashboard.
        """
    )

    st.markdown("---")

    st.caption(
        "BTech Data Science Engineering Project"
    )