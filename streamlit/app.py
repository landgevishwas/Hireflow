from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor

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
from concurrent.futures import ThreadPoolExecutor, as_completed


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
       GLOBAL
       ===================================================================== */

    .stApp {
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
    }

    .main {
        background-color: #ffffff !important;
    }

    .main .block-container {
        max-width: 1250px !important;
        padding-top: 2.5rem !important;
        padding-bottom: 3rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
    }


    /* =====================================================================
       TITLE
       ===================================================================== */

    .hireflow-title {
        color: #000000 !important;
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
        color: #333333 !important;
        font-size: 17px !important;
        font-weight: 400 !important;
        line-height: 1.5 !important;
        margin: 0 0 34px 0 !important;
        padding: 0 !important;
        opacity: 1 !important;
    }


    /* =====================================================================
       HEADINGS
       ===================================================================== */

    h1,
    h2,
    h3,
    h4,
    h5,
    h6 {
        color: #000000 !important;
    }

    .section-title {
        color: #000000 !important;
        font-size: 27px !important;
        font-weight: 750 !important;
        line-height: 1.3 !important;
        margin-top: 26px !important;
        margin-bottom: 14px !important;
    }


    /* =====================================================================
       TEXT
       ===================================================================== */

    .main p,
    .main span,
    .main li {
        color: #000000 !important;
    }

    .main [data-testid="stMarkdownContainer"] p {
        color: #000000 !important;
    }

    .main [data-testid="stCaptionContainer"],
    .main [data-testid="stCaptionContainer"] p {
        color: #555555 !important;
    }


    /* =====================================================================
       LABELS
       ===================================================================== */

    .main label,
    .main [data-testid="stWidgetLabel"],
    .main [data-testid="stWidgetLabel"] p {
        color: #000000 !important;
        font-weight: 600 !important;
    }


    /* =====================================================================
       TEXT INPUT
       ===================================================================== */

    .main input {
        background-color: #ffffff !important;
        color: #000000 !important;

        border: 1.5px solid #9ca3af !important;
        border-radius: 8px !important;

        box-shadow: none !important;
    }

    .main input:hover {
        border-color: #555555 !important;
    }

    .main input:focus {
        background-color: #ffffff !important;
        color: #000000 !important;

        border: 2px solid #000000 !important;

        box-shadow: none !important;
    }

    .main input::placeholder {
        color: #666666 !important;
        opacity: 1 !important;
    }


    /* =====================================================================
       TEXT AREA
       ===================================================================== */

    .main textarea {
        background-color: #ffffff !important;
        color: #000000 !important;

        border: 1.5px solid #9ca3af !important;
        border-radius: 8px !important;

        box-shadow: none !important;
    }

    .main textarea:hover {
        border-color: #555555 !important;
    }

    .main textarea:focus {
        background-color: #ffffff !important;
        color: #000000 !important;

        border: 2px solid #000000 !important;

        box-shadow: none !important;
    }

    .main textarea::placeholder {
        color: #666666 !important;
        opacity: 1 !important;
    }


    /* =====================================================================
       NUMBER INPUT
       ===================================================================== */

    .main [data-testid="stNumberInput"] {
        border-radius: 8px !important;
    }

    .main [data-testid="stNumberInput"] input {
        background-color: #ffffff !important;
        color: #000000 !important;

        border: 1.5px solid #9ca3af !important;
    }


    /* =====================================================================
       FILE UPLOADER
       ===================================================================== */

    [data-testid="stFileUploader"] {
        background-color: #ffffff !important;

        border: 1.5px solid #9ca3af !important;
        border-radius: 10px !important;

        padding: 8px !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: #ffffff !important;

        border: 1.5px dashed #9ca3af !important;
        border-radius: 8px !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #000000 !important;
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: #000000 !important;
    }

    [data-testid="stFileUploader"] label {
        color: #000000 !important;
    }


    /* =====================================================================
       BUTTONS
       ===================================================================== */

    .stButton > button {
        background-color: #000000 !important;
        color: #ffffff !important;

        border: 1.5px solid #000000 !important;
        border-radius: 8px !important;

        font-weight: 700 !important;

        min-height: 42px !important;

        padding-left: 20px !important;
        padding-right: 20px !important;

        box-shadow: none !important;
    }

    .stButton > button:hover {
        background-color: #333333 !important;
        color: #ffffff !important;
        border-color: #000000 !important;
    }

    .stButton > button p,
    .stButton > button span {
        color: #ffffff !important;
    }


    /* =====================================================================
       METRIC CARDS
       ===================================================================== */

    [data-testid="stMetric"] {
        background-color: #ffffff !important;

        border: 1.5px solid #c7c7c7 !important;
        border-radius: 10px !important;

        padding: 14px !important;

        box-shadow: none !important;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        color: #555555 !important;
    }

    [data-testid="stMetricValue"] {
        color: #000000 !important;
    }


    /* =====================================================================
       CANDIDATE CARDS
       ===================================================================== */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;

        border: 1.5px solid #c7c7c7 !important;
        border-radius: 12px !important;

        box-shadow: none !important;
    }


    /* =====================================================================
       EXPANDERS
       ===================================================================== */

    [data-testid="stExpander"] {
        background-color: #ffffff !important;

        border: 1.5px solid #c7c7c7 !important;
        border-radius: 9px !important;
    }

    [data-testid="stExpander"] summary {
        color: #000000 !important;
    }

    [data-testid="stExpander"] summary span {
        color: #000000 !important;
    }


    /* =====================================================================
       SELECTBOX
       ===================================================================== */

    .main [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #000000 !important;

        border: 1.5px solid #9ca3af !important;
        border-radius: 8px !important;
    }

    .main [data-baseweb="select"] * {
        color: #000000 !important;
    }


    /* =====================================================================
       RADIO
       ===================================================================== */

    .main [data-testid="stRadio"] label,
    .main [data-testid="stRadio"] label p {
        color: #000000 !important;
    }


    /* =====================================================================
       DIVIDER
       ===================================================================== */

    .main hr {
        border: none !important;
        border-top: 1px solid #d0d0d0 !important;
    }


    /* =====================================================================
       ALERTS
       ===================================================================== */

    [data-testid="stAlert"] {
        border: 1px solid #c7c7c7 !important;
        border-radius: 8px !important;
    }


    /* =====================================================================
       PROGRESS BAR
       ===================================================================== */

    [data-testid="stProgress"] {
        border: 1px solid #c7c7c7 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }


    /* =====================================================================
       SIDEBAR
       ===================================================================== */

    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"] {
        background-color: #ffffff !important;
        color: #000000 !important;

        border-right: 1.5px solid #d0d0d0 !important;
    }

    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #000000 !important;
    }


    /* =====================================================================
       SIDEBAR STATUS
       ===================================================================== */

    .status-ready {
        background-color: #ffffff !important;

        color: #000000 !important;

        border: 1.5px solid #198754 !important;
        border-radius: 9px !important;

        padding: 12px 14px;

        text-align: center;

        font-weight: 700;

        margin-top: 10px;
        margin-bottom: 12px;
    }


    /* =====================================================================
       SIDEBAR METRICS
       ===================================================================== */

    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background-color: #ffffff !important;

        border: 1.5px solid #c7c7c7 !important;
        border-radius: 10px !important;

        padding: 14px !important;
    }

    [data-testid="stSidebar"] [data-testid="stMetricLabel"],
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] p {
        color: #555555 !important;
    }

    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #000000 !important;
    }


    /* =====================================================================
       SIDEBAR RADIO
       ===================================================================== */

    [data-testid="stSidebar"] [data-testid="stRadio"] label,
    [data-testid="stSidebar"] [data-testid="stRadio"] label p {
        color: #000000 !important;
    }


    /* =====================================================================
       MOBILE
       ===================================================================== */

    @media (max-width: 768px) {

        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 1.5rem !important;
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

if "candidate_sources" not in st.session_state:
    st.session_state.candidate_sources = {}


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
# HELPERS
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


def resolve_resume_path(
    candidate_id: str,
    source: str = "",
) -> Path | None:
    """Resolve the candidate PDF from indexed source or candidate ID."""

    if source:
        source_path = Path(source)
        if source_path.exists():
            return source_path

        relative_path = Path(settings.resume_dir) / source_path.name
        if relative_path.exists():
            return relative_path

    exact_path = Path(settings.resume_dir) / f"{candidate_id}.pdf"
    if exact_path.exists():
        return exact_path

    resume_dir = Path(settings.resume_dir)
    if not resume_dir.exists():
        return None

    normalized_id = (
        candidate_id.lower().replace(" ", "_").replace("-", "_")
    )

    for pdf_path in resume_dir.glob("*.pdf"):
        normalized_name = (
            pdf_path.stem.lower().replace(" ", "_").replace("-", "_")
        )
        if normalized_name == normalized_id:
            return pdf_path

    return None


def get_candidate(
    candidate_id: str,
    source: str = "",
):
    """Load a candidate from cache or parse the source resume PDF."""

    cached = st.session_state.candidate_cache.get(candidate_id)
    if cached is not None:
        return cached

    resume_path = resolve_resume_path(candidate_id, source)
    if resume_path is None:
        return None

    try:
        candidate = parse_candidate_pdf(
            resume_path,
            candidate_id=candidate_id,
        )
    except TypeError:
        try:
            candidate = parse_candidate_pdf(resume_path)
        except Exception as exc:
            st.session_state[f"parse_error_{candidate_id}"] = str(exc)
            return None
    except Exception as exc:
        st.session_state[f"parse_error_{candidate_id}"] = str(exc)
        return None

    st.session_state.candidate_cache[candidate_id] = candidate
    st.session_state.pop(f"parse_error_{candidate_id}", None)
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
        "source": str(getattr(result, "source", "")),
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
# SEARCH PAGE
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
    # SEARCH CANDIDATES
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
    # SEARCH EXECUTION
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

        try:

            index_size = (
                indexer.semantic_index_size
            )

        except Exception:

            index_size = 0

        if index_size == 0:

            st.warning(
                "No resumes are indexed yet. "
                "Upload and process resumes first."
            )

            st.stop()

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
        candidates_to_parse = []

        # Prepare retrieval metadata first. Candidates already cached in this
        # Streamlit session are reused without another Gemini call.
        for result in hybrid_results:
            result_dict = result_to_dict(result)
            candidate_id = result_dict["candidate_id"]

            retrieval_map[candidate_id] = result_dict

            source = result_dict.get("source", "")
            if not source:
                source = str(
                    Path(settings.resume_dir) / f"{candidate_id}.pdf"
                )

            st.session_state.candidate_sources[candidate_id] = source

            cached = st.session_state.candidate_cache.get(candidate_id)
            if cached is not None:
                parsed_candidates.append(cached)
            else:
                resume_path = resolve_resume_path(candidate_id, source)
                if resume_path is not None:
                    candidates_to_parse.append(
                        (candidate_id, resume_path)
                    )

        def _parse_candidate_for_search(item):
            candidate_id, resume_path = item
            try:
                try:
                    candidate = parse_candidate_pdf(
                        resume_path,
                        candidate_id=candidate_id,
                    )
                except TypeError:
                    candidate = parse_candidate_pdf(resume_path)
                return candidate_id, candidate, None
            except Exception as exc:
                return candidate_id, None, str(exc)

        # Gemini structured parsing is the slowest part of search. Parse the
        # retrieved candidates concurrently instead of making 10 sequential
        # Gemini requests. The cap avoids creating excessive API concurrency.
        if candidates_to_parse:
            worker_count = min(5, len(candidates_to_parse))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        _parse_candidate_for_search,
                        item,
                    )
                    for item in candidates_to_parse
                ]

                for future in as_completed(futures):
                    candidate_id, candidate, error = future.result()
                    if candidate is not None:
                        st.session_state.candidate_cache[candidate_id] = candidate
                        parsed_candidates.append(candidate)
                    elif error:
                        st.session_state[f"parse_error_{candidate_id}"] = error

        # --------------------------------------------------------------------
        # FILTER
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

            candidates_for_reranking = parsed_candidates

        # --------------------------------------------------------------------
        # RE-RANK
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

        if reranked:

            st.session_state.search_results = reranked

        else:

            st.session_state.search_results = hybrid_results

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
            # CARD
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
                # AI EVALUATION
                # =============================================================

                if st.button(
                    "View AI Evaluation",
                    key=f"evaluate_{candidate_id}",
                ):

                    # Retry parsing on demand if search-time parsing failed.
                    if candidate is None:
                        source = st.session_state.candidate_sources.get(
                            candidate_id,
                            "",
                        )

                        with st.spinner("Loading candidate resume..."):
                            candidate = get_candidate(
                                candidate_id=candidate_id,
                                source=source,
                            )

                    if candidate is None:
                        parse_error = st.session_state.get(
                            f"parse_error_{candidate_id}",
                            "",
                        )

                        st.error(
                            "Candidate resume could not be loaded or parsed."
                        )

                        if parse_error:
                            st.caption(f"Parser error: {parse_error}")

                    else:
                        job = st.session_state.last_job

                        if job is None:
                            st.error(
                                "Job information is unavailable. "
                                "Please perform a new search."
                            )

                        else:
                            with st.spinner(
                                "Running structured AI evaluation..."
                            ):
                                try:
                                    evaluator = get_evaluator()

                                    evaluation = evaluator.evaluate_candidate(
                                        candidate,
                                        job,
                                    )

                                    st.session_state[
                                        f"evaluation_{candidate_id}"
                                    ] = evaluation

                                    st.session_state[
                                        "candidates_viewed"
                                    ] += 1

                                except Exception as exc:
                                    st.error("AI evaluation failed.")
                                    st.exception(exc)

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
    # RECRUITER PREFERENCES
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
    # ADD PREFERENCE
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