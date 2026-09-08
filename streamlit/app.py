from __future__ import annotations

import sys
from pathlib import Path

# ============================================================================
# PROJECT PATH
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# IMPORTS
# ============================================================================

import streamlit as st

from config import get_settings
from schemas import JobDescription, SkillRequirement

from core.hybrid_indexer import HybridIndexer
from core.parsing import parse_candidate_pdf
from core.filters import apply_filters
from core.re_ranker import rerank_candidates
from core.evaluator import CandidateEvaluator
from core.memory_rag import RecruiterMemoryStore


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="HireFlow — AI-Powered Recruitment",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown(
    """
    <style>

    /* =====================================================================
       MAIN APPLICATION
       ===================================================================== */

    .stApp {
        background-color: #0b0f17 !important;
    }

    .main {
        background-color: #0b0f17 !important;
    }

    .main .block-container {
        max-width: 1250px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }


    /* =====================================================================
       MAIN TITLE
       ===================================================================== */

    .hireflow-title {
        color: #ffffff !important;
        font-size: 42px !important;
        font-weight: 800 !important;
        line-height: 1.2 !important;
        letter-spacing: -0.8px !important;
        margin: 0 0 8px 0 !important;
        padding: 0 !important;
        opacity: 1 !important;
        text-shadow: none !important;
    }


    /* =====================================================================
       SUBTITLE
       ===================================================================== */

    .hireflow-subtitle {
        color: #cbd5e1 !important;
        font-size: 17px !important;
        font-weight: 400 !important;
        line-height: 1.5 !important;
        margin: 0 0 34px 0 !important;
        padding: 0 !important;
        opacity: 1 !important;
    }


    /* =====================================================================
       SECTION TITLES
       ===================================================================== */

    .section-title {
        color: #ffffff !important;
        font-size: 27px !important;
        font-weight: 750 !important;
        line-height: 1.3 !important;
        margin-top: 26px !important;
        margin-bottom: 14px !important;
        opacity: 1 !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }


    /* =====================================================================
       NORMAL TEXT
       ===================================================================== */

    .main p,
    .main span,
    .main label,
    .main div {
        color: inherit;
    }

    .main [data-testid="stMarkdownContainer"] p {
        color: #e5e7eb;
    }

    .main [data-testid="stCaptionContainer"] {
        color: #aeb8c7 !important;
    }


    /* =====================================================================
       STREAMLIT INPUT LABELS
       ===================================================================== */

    .main label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
    }

    .main [data-testid="stWidgetLabel"] p {
        color: #cbd5e1 !important;
    }


    /* =====================================================================
       TEXT INPUT / TEXT AREA / NUMBER INPUT
       ===================================================================== */

    .main input,
    .main textarea {
        color: #111827 !important;
        background-color: #f8fafc !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
    }

    .main input::placeholder,
    .main textarea::placeholder {
        color: #64748b !important;
        opacity: 1 !important;
    }

    .main input:focus,
    .main textarea:focus {
        border-color: #64748b !important;
        box-shadow: 0 0 0 1px #64748b !important;
    }


    /* =====================================================================
       FILE UPLOADER
       ===================================================================== */

    [data-testid="stFileUploader"] {
        background-color: #f8fafc !important;
        border-radius: 10px !important;
    }

    [data-testid="stFileUploader"] label {
        color: #334155 !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: #f8fafc !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: #475569 !important;
    }


    /* =====================================================================
       BUTTONS
       ===================================================================== */

    .stButton > button {
        border-radius: 8px !important;
        font-weight: 700 !important;
        min-height: 42px !important;
        padding-left: 18px !important;
        padding-right: 18px !important;
    }


    /* =====================================================================
       METRIC CARDS
       ===================================================================== */

    [data-testid="stMetric"] {
        background-color: #151b26 !important;
        border: 1px solid #293241 !important;
        border-radius: 12px !important;
        padding: 14px !important;
    }

    [data-testid="stMetricLabel"] {
        color: #aeb8c7 !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }


    /* =====================================================================
       RESULT CARDS
       ===================================================================== */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #121823 !important;
        border: 1px solid #293241 !important;
        border-radius: 14px !important;
    }


    /* =====================================================================
       EXPANDERS
       ===================================================================== */

    [data-testid="stExpander"] {
        background-color: #121823 !important;
        border: 1px solid #293241 !important;
        border-radius: 10px !important;
    }

    [data-testid="stExpander"] summary {
        color: #e5e7eb !important;
    }


    /* =====================================================================
       DIVIDERS
       ===================================================================== */

    .main hr {
        border-color: #293241 !important;
    }


    /* =====================================================================
       SIDEBAR
       ===================================================================== */

    [data-testid="stSidebar"] {
        background-color: #eef2f7 !important;
    }

    [data-testid="stSidebar"] * {
        color: #1f2937;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #1f2937 !important;
    }


    /* Sidebar status */

    .status-ready {
        background-color: #123c29;
        color: #67e8a5 !important;
        padding: 12px 14px;
        border-radius: 9px;
        text-align: center;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 12px;
    }


    /* =====================================================================
       SIDEBAR METRICS
       ===================================================================== */

    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background-color: #151a24 !important;
        border: 1px solid #252c39 !important;
        border-radius: 11px !important;
    }

    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #aab3c2 !important;
    }

    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #e5e7eb !important;
    }


    /* =====================================================================
       MOBILE
       ===================================================================== */

    @media (max-width: 768px) {

        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1.5rem;
        }

        .hireflow-title {
            font-size: 31px !important;
        }

        .hireflow-subtitle {
            font-size: 15px !important;
        }

        .section-title {
            font-size: 23px !important;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# SETTINGS
# ============================================================================

settings = get_settings()


# ============================================================================
# SESSION STATE
# ============================================================================

if "search_count" not in st.session_state:
    st.session_state.search_count = 0

if "candidates_viewed" not in st.session_state:
    st.session_state.candidates_viewed = 0

if "search_results" not in st.session_state:
    st.session_state.search_results = []

if "selected_candidate" not in st.session_state:
    st.session_state.selected_candidate = None

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if "last_job" not in st.session_state:
    st.session_state.last_job = None

if "candidate_cache" not in st.session_state:
    st.session_state.candidate_cache = {}

if "uploaded_candidates" not in st.session_state:
    st.session_state.uploaded_candidates = []


# ============================================================================
# SERVICES
# ============================================================================

@st.cache_resource
def get_indexer():
    return HybridIndexer()


@st.cache_resource
def get_evaluator():
    return CandidateEvaluator()


@st.cache_resource
def get_memory_store():
    return RecruiterMemoryStore()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_skill_list(value: str) -> list[str]:

    if not value:
        return []

    skills = []

    for item in value.split(","):

        skill = item.strip()

        if skill:
            skills.append(skill)

    return list(dict.fromkeys(skills))


def build_job_description(
    query: str,
    description: str,
    required_skills: list[str],
    minimum_experience: float,
) -> JobDescription:

    skill_requirements = [
        SkillRequirement(
            name=skill,
            requirement_type="required",
        )
        for skill in required_skills
    ]

    return JobDescription(
        job_id="streamlit_search",
        title=query.strip(),
        company="",
        location="",
        employment_type="",
        work_mode="",
        salary_range="",
        responsibilities=[],
        skill_requirements=skill_requirements,
        required_experience_years_min=(
            minimum_experience
            if minimum_experience > 0
            else None
        ),
        required_experience_years_max=None,
        education_requirements=[],
        certifications=[],
        raw_text=description.strip(),
    )


def get_candidate(candidate_id: str):

    if candidate_id in st.session_state.candidate_cache:
        return st.session_state.candidate_cache[candidate_id]

    resume_path = (
        Path(settings.resume_dir)
        / f"{candidate_id}.pdf"
    )

    if not resume_path.exists():
        return None

    try:

        candidate = parse_candidate_pdf(
            resume_path,
            candidate_id=candidate_id,
        )

    except TypeError:

        try:
            candidate = parse_candidate_pdf(
                resume_path
            )
        except Exception:
            return None

    except Exception:
        return None

    st.session_state.candidate_cache[
        candidate_id
    ] = candidate

    return candidate


def result_to_dict(result) -> dict:

    if hasattr(result, "to_dict"):
        return result.to_dict()

    return {
        "candidate_id": getattr(
            result,
            "candidate_id",
            "",
        ),
        "hybrid_score": float(
            getattr(
                result,
                "hybrid_score",
                0.0,
            )
        ),
        "semantic_score": float(
            getattr(
                result,
                "semantic_score",
                0.0,
            )
        ),
        "keyword_score": float(
            getattr(
                result,
                "keyword_score",
                0.0,
            )
        ),
    }


def score_display(score: float) -> str:

    if score <= 1:
        return f"{score * 100:.1f}%"

    return f"{score:.1f}"


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:

    st.markdown(
        "## HireFlow"
    )

    st.markdown(
        """
        <div class="status-ready">
            Status: Ready
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.metric(
        "Searches this session",
        st.session_state.search_count,
    )

    st.metric(
        "Candidates viewed",
        st.session_state.candidates_viewed,
    )

    st.divider()

    st.markdown(
        "### Navigate"
    )

    page = st.radio(
        "",
        [
            "Search Candidates",
            "Memory & Evaluation",
        ],
        label_visibility="collapsed",
    )


# ============================================================================
# MAIN HEADER
# ============================================================================

st.markdown(
    """
    <div class="hireflow-title">
        HireFlow — AI-Powered Recruitment
    </div>

    <div class="hireflow-subtitle">
        Intelligent Candidate Search and Evaluation
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# SEARCH CANDIDATES PAGE
# ============================================================================

if page == "Search Candidates":

    # ========================================================================
    # UPLOAD RESUMES
    # ========================================================================

    st.markdown(
        """
        <div class="section-title">
            Upload Resumes
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload resume PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more candidate resumes.",
    )

    if uploaded_files:

        if st.button(
            "Process & Index Resumes",
            type="primary",
            key="process_resumes",
        ):

            settings.resume_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            indexer = get_indexer()

            successful = 0
            failed = 0

            progress = st.progress(
                0,
                text="Processing resumes...",
            )

            total = len(uploaded_files)

            for position, uploaded_file in enumerate(
                uploaded_files,
                start=1,
            ):

                try:

                    destination = (
                        Path(settings.resume_dir)
                        / uploaded_file.name
                    )

                    destination.write_bytes(
                        uploaded_file.getvalue()
                    )

                    candidate_id = destination.stem

                    # Index resume.
                    indexer.add_resume(
                        destination,
                        candidate_id=candidate_id,
                    )

                    st.session_state.candidate_cache.pop(
                        candidate_id,
                        None,
                    )

                    if candidate_id not in (
                        st.session_state.uploaded_candidates
                    ):
                        st.session_state.uploaded_candidates.append(
                            candidate_id
                        )

                    successful += 1

                except Exception as exc:

                    failed += 1

                    st.error(
                        f"Could not process "
                        f"{uploaded_file.name}: {exc}"
                    )

                progress.progress(
                    position / total,
                    text=f"Processing {position}/{total}",
                )

            progress.empty()

            if successful:

                st.success(
                    f"{successful} resume(s) "
                    "uploaded and indexed successfully."
                )

            if failed:

                st.warning(
                    f"{failed} resume(s) failed."
                )


    # ========================================================================
    # SEARCH SECTION
    # ========================================================================

    st.markdown(
        """
        <div class="section-title">
            Search Candidates
        </div>
        """,
        unsafe_allow_html=True,
    )

    query = st.text_input(
        "Job title or search query",
        placeholder="e.g. staff accountant with QuickBooks",
        key="job_query",
    )

    job_description = st.text_area(
        "Job description (optional — improves AI evaluation)",
        placeholder="Paste the complete job description here...",
        height=130,
        key="job_description",
    )

    col1, col2 = st.columns(
        [1.4, 1]
    )

    with col1:

        required_skills_text = st.text_input(
            "Required skills (comma-separated)",
            placeholder="Excel, QuickBooks, GAAP, SAP",
            key="required_skills",
        )

    with col2:

        minimum_experience = st.number_input(
            "Minimum years of experience",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=0.5,
            key="minimum_experience",
        )

    search_button = st.button(
        "Search",
        type="primary",
        key="search_candidates",
    )


    # ========================================================================
    # SEARCH
    # ========================================================================

    if search_button:

        if not query.strip():

            st.warning(
                "Please enter a job title or search query."
            )

            st.stop()

        required_skills = clean_skill_list(
            required_skills_text
        )

        job = build_job_description(
            query=query,
            description=job_description,
            required_skills=required_skills,
            minimum_experience=minimum_experience,
        )

        indexer = get_indexer()

        # --------------------------------------------------------------------
        # INDEX CHECK
        # --------------------------------------------------------------------

        try:
            index_size = indexer.semantic_index_size
        except Exception:
            index_size = 0

        if index_size == 0:

            st.warning(
                "No resumes are indexed yet. "
                "Upload and process resumes first."
            )

            st.stop()

        # --------------------------------------------------------------------
        # HYBRID SEARCH
        # --------------------------------------------------------------------

        search_query = query.strip()

        if job_description.strip():

            search_query += (
                "\n"
                + job_description.strip()
            )

        with st.spinner(
            "Searching candidates..."
        ):

            try:

                hybrid_results = indexer.search(
                    search_query,
                    top_k=settings.final_top_k,
                )

            except Exception as exc:

                st.error(
                    f"Candidate search failed: {exc}"
                )

                st.stop()

        # --------------------------------------------------------------------
        # PARSE CANDIDATES
        # --------------------------------------------------------------------

        parsed_candidates = []

        retrieval_map = {}

        for result in hybrid_results:

            result_dict = result_to_dict(
                result
            )

            candidate_id = result_dict[
                "candidate_id"
            ]

            retrieval_map[
                candidate_id
            ] = result_dict

            candidate = get_candidate(
                candidate_id
            )

            if candidate is not None:

                parsed_candidates.append(
                    candidate
                )

        # --------------------------------------------------------------------
        # HARD FILTERS
        # --------------------------------------------------------------------

        filter_results = []

        for candidate in parsed_candidates:

            try:

                filter_result = apply_filters(
                    candidate,
                    job,
                )

                filter_results.append(
                    filter_result
                )

            except Exception:
                continue

        eligible_ids = {
            item.candidate_id
            for item in filter_results
            if item.eligible
        }

        if eligible_ids:

            candidates_for_reranking = [
                candidate
                for candidate in parsed_candidates
                if candidate.candidate_id
                in eligible_ids
            ]

        else:

            candidates_for_reranking = (
                parsed_candidates
            )

        # --------------------------------------------------------------------
        # EXPLAINABLE RE-RANKING
        # --------------------------------------------------------------------

        reranked = []

        if candidates_for_reranking:

            try:

                reranked = rerank_candidates(
                    candidates_for_reranking,
                    job,
                    retrieval_scores=retrieval_map,
                )

            except TypeError:

                try:

                    reranked = rerank_candidates(
                        candidates_for_reranking,
                        job,
                    )

                except Exception as exc:

                    st.warning(
                        f"Re-ranking unavailable: {exc}"
                    )

            except Exception as exc:

                st.warning(
                    f"Re-ranking unavailable: {exc}"
                )

        # --------------------------------------------------------------------
        # SAVE RESULTS
        # --------------------------------------------------------------------

        if reranked:

            st.session_state.search_results = (
                reranked
            )

        else:

            st.session_state.search_results = (
                hybrid_results
            )

        st.session_state.last_query = query
        st.session_state.last_job = job
        st.session_state.search_count += 1

        st.rerun()


    # ========================================================================
    # RESULTS
    # ========================================================================

    results = st.session_state.search_results

    if results:

        st.divider()

        st.subheader(
            f"Candidate Results ({len(results)})"
        )

        for position, result in enumerate(
            results,
            start=1,
        ):

            # =================================================================
            # RESULT DATA
            # =================================================================

            if hasattr(
                result,
                "candidate_id",
            ):

                candidate_id = result.candidate_id

                final_score = float(
                    getattr(
                        result,
                        "final_score",
                        0.0,
                    )
                )

                hybrid_score = float(
                    getattr(
                        result,
                        "hybrid_score",
                        0.0,
                    )
                )

                experience_score = float(
                    getattr(
                        result,
                        "experience_score",
                        0.0,
                    )
                )

                required_score = float(
                    getattr(
                        result,
                        "required_skills_score",
                        0.0,
                    )
                )

                preferred_score = float(
                    getattr(
                        result,
                        "preferred_skills_score",
                        0.0,
                    )
                )

                matched_required = list(
                    getattr(
                        result,
                        "matched_required_skills",
                        (),
                    )
                )

                missing_required = list(
                    getattr(
                        result,
                        "missing_required_skills",
                        (),
                    )
                )

                strengths = list(
                    getattr(
                        result,
                        "strengths",
                        (),
                    )
                )

                concerns = list(
                    getattr(
                        result,
                        "concerns",
                        (),
                    )
                )

            else:

                result_dict = result_to_dict(
                    result
                )

                candidate_id = result_dict[
                    "candidate_id"
                ]

                final_score = result_dict[
                    "hybrid_score"
                ]

                hybrid_score = result_dict[
                    "hybrid_score"
                ]

                experience_score = 0.0
                required_score = 0.0
                preferred_score = 0.0

                matched_required = []
                missing_required = []

                strengths = []
                concerns = []


            # =================================================================
            # CANDIDATE
            # =================================================================

            candidate = (
                st.session_state.candidate_cache.get(
                    candidate_id
                )
            )

            candidate_name = candidate_id
            candidate_location = ""
            candidate_experience = None

            if candidate:

                if candidate.name.strip():
                    candidate_name = candidate.name

                candidate_location = candidate.location

                candidate_experience = (
                    candidate.total_experience_years
                )


            # =================================================================
            # RESULT CARD
            # =================================================================

            with st.container(border=True):

                left, right = st.columns(
                    [5, 1]
                )

                with left:

                    st.markdown(
                        f"### {position}. {candidate_name}"
                    )

                    if candidate_location:

                        st.caption(
                            f"📍 {candidate_location}"
                        )

                    if candidate_experience is not None:

                        st.caption(
                            f"Experience: "
                            f"{candidate_experience:.1f} years"
                        )

                    if matched_required:

                        st.write(
                            "**Matched skills:** "
                            + ", ".join(
                                matched_required
                            )
                        )

                    if missing_required:

                        st.warning(
                            "**Missing required:** "
                            + ", ".join(
                                missing_required
                            )
                        )

                with right:

                    st.metric(
                        "Match Score",
                        score_display(
                            final_score
                        ),
                    )


                # =============================================================
                # SCORE BREAKDOWN
                # =============================================================

                with st.expander(
                    "Score Breakdown & Explainability"
                ):

                    c1, c2, c3, c4 = st.columns(4)

                    with c1:

                        st.metric(
                            "Hybrid",
                            score_display(
                                hybrid_score
                            ),
                        )

                    with c2:

                        st.metric(
                            "Experience",
                            score_display(
                                experience_score
                            ),
                        )

                    with c3:

                        st.metric(
                            "Required Skills",
                            score_display(
                                required_score
                            ),
                        )

                    with c4:

                        st.metric(
                            "Preferred Skills",
                            score_display(
                                preferred_score
                            ),
                        )

                    if strengths:

                        st.markdown(
                            "**Strengths**"
                        )

                        for item in strengths:

                            st.write(
                                f"✓ {item}"
                            )

                    if concerns:

                        st.markdown(
                            "**Concerns**"
                        )

                        for item in concerns:

                            st.write(
                                f"• {item}"
                            )


                # =============================================================
                # AI EVALUATION BUTTON
                # =============================================================

                if st.button(
                    "View AI Evaluation",
                    key=f"evaluate_{candidate_id}",
                ):

                    if candidate is None:

                        st.error(
                            "Candidate resume could not be parsed."
                        )

                    else:

                        job = st.session_state.last_job

                        if job is None:

                            st.error(
                                "Job information is unavailable."
                            )

                        else:

                            with st.spinner(
                                "Running structured AI evaluation..."
                            ):

                                try:

                                    evaluator = (
                                        get_evaluator()
                                    )

                                    evaluation = (
                                        evaluator.evaluate(
                                            candidate,
                                            job,
                                        )
                                    )

                                    st.session_state[
                                        f"evaluation_{candidate_id}"
                                    ] = evaluation

                                    st.session_state[
                                        "candidates_viewed"
                                    ] += 1

                                except Exception as exc:

                                    st.error(
                                        "AI evaluation failed: "
                                        f"{exc}"
                                    )


                # =============================================================
                # AI EVALUATION RESULT
                # =============================================================

                evaluation = st.session_state.get(
                    f"evaluation_{candidate_id}"
                )

                if evaluation:

                    st.divider()

                    st.markdown(
                        "#### AI Evaluation"
                    )

                    eval_col1, eval_col2 = st.columns(
                        [1, 3]
                    )

                    with eval_col1:

                        st.metric(
                            "AI Score",
                            f"{evaluation.overall_score:.0f}/100",
                        )

                        st.write(
                            "**Recommendation**"
                        )

                        st.info(
                            evaluation.recommendation
                        )

                    with eval_col2:

                        if evaluation.reasoning:

                            st.write(
                                "**Reasoning**"
                            )

                            st.write(
                                evaluation.reasoning
                            )

                    if evaluation.strengths:

                        st.write(
                            "**Strengths**"
                        )

                        for item in evaluation.strengths:

                            st.write(
                                f"✓ {item}"
                            )

                    if evaluation.concerns:

                        st.write(
                            "**Concerns**"
                        )

                        for item in evaluation.concerns:

                            st.write(
                                f"• {item}"
                            )

                    if evaluation.missing_requirements:

                        st.warning(
                            "Missing requirements: "
                            + ", ".join(
                                evaluation.missing_requirements
                            )
                        )

                    if evaluation.evidence:

                        with st.expander(
                            "Evidence"
                        ):

                            for evidence in evaluation.evidence:

                                st.markdown(
                                    f"**{evidence.requirement}**"
                                )

                                st.write(
                                    evidence.evidence
                                )

                                st.caption(
                                    f"Source: {evidence.source}"
                                )


# ============================================================================
# MEMORY & EVALUATION PAGE
# ============================================================================

else:

    st.markdown(
        """
        <div class="section-title">
            Memory & Evaluation
        </div>
        """,
        unsafe_allow_html=True,
    )

    memory_store = get_memory_store()

    memory = memory_store.load()

    # ========================================================================
    # MEMORY
    # ========================================================================

    st.subheader(
        "Recruiter Preferences"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "**Preferred Skills**"
        )

        if memory.preferred_skills:

            for skill in memory.preferred_skills:

                st.write(
                    f"✓ {skill}"
                )

        else:

            st.caption(
                "No preferred skills saved."
            )

        st.write(
            "**Preferred Titles**"
        )

        if memory.preferred_titles:

            for title in memory.preferred_titles:

                st.write(
                    f"✓ {title}"
                )

        else:

            st.caption(
                "No preferred titles saved."
            )

    with col2:

        st.write(
            "**Excluded Skills**"
        )

        if memory.excluded_skills:

            for skill in memory.excluded_skills:

                st.write(
                    f"• {skill}"
                )

        else:

            st.caption(
                "No excluded skills saved."
            )

        st.write(
            "**Preferred Industries**"
        )

        if memory.preferred_industries:

            for industry in memory.preferred_industries:

                st.write(
                    f"✓ {industry}"
                )

        else:

            st.caption(
                "No preferred industries saved."
            )


    # ========================================================================
    # ADD MEMORY
    # ========================================================================

    st.divider()

    st.subheader(
        "Add Recruiter Preference"
    )

    preference_type = st.selectbox(
        "Preference type",
        [
            "Preferred Skill",
            "Excluded Skill",
            "Preferred Title",
            "Preferred Industry",
            "Note",
        ],
    )

    preference_value = st.text_input(
        "Value",
        placeholder="e.g. QuickBooks",
    )

    if st.button(
        "Save Preference",
        type="primary",
    ):

        if not preference_value.strip():

            st.warning(
                "Please enter a value."
            )

        else:

            value = preference_value.strip()

            if preference_type == "Preferred Skill":

                memory_store.add_preferred_skill(
                    value
                )

            elif preference_type == "Excluded Skill":

                memory_store.add_excluded_skill(
                    value
                )

            elif preference_type == "Preferred Title":

                memory_store.add_preferred_title(
                    value
                )

            elif preference_type == "Preferred Industry":

                memory_store.add_preferred_industry(
                    value
                )

            elif preference_type == "Note":

                memory_store.add_note(
                    value
                )

            st.success(
                "Preference saved."
            )

            st.rerun()


    # ========================================================================
    # SYSTEM INFORMATION
    # ========================================================================

    st.divider()

    st.subheader(
        "System Information"
    )

    indexer = get_indexer()

    try:
        semantic_size = indexer.semantic_index_size
    except Exception:
        semantic_size = 0

    try:
        keyword_size = indexer.keyword_index_size
    except Exception:
        keyword_size = 0

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "FAISS Candidates",
            semantic_size,
        )

    with c2:

        st.metric(
            "BM25 Candidates",
            keyword_size,
        )

    with c3:

        st.metric(
            "Searches",
            st.session_state.search_count,
        )

    st.caption(
        "Recruiter memory is advisory only. "
        "It does not override explicit job requirements "
        "or hard candidate filters."
    )