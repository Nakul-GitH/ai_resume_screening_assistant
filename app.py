import os
import tempfile
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# ============================================================
# 1. Application Configuration
# ============================================================

st.set_page_config(
    page_title="AI Resume Screening Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. Custom Styling
# ============================================================

st.markdown(
    """
    <style>
        .main {
            background-color: #f7f8fa;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        .app-header {
            background: linear-gradient(135deg, #172554, #1e3a8a);
            padding: 2rem 2.5rem;
            border-radius: 14px;
            margin-bottom: 1.5rem;
            color: white;
        }

        .app-header h1 {
            margin: 0;
            font-size: 2.2rem;
            font-weight: 700;
        }

        .app-header p {
            margin: 0.6rem 0 0 0;
            font-size: 1rem;
            opacity: 0.9;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 700;
            color: #172554;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
        }

        .candidate-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        }

        .metric-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        }

        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #172554;
        }

        .metric-label {
            font-size: 0.85rem;
            color: #6b7280;
            margin-top: 0.2rem;
        }

        .info-box {
            background: #eff6ff;
            border-left: 4px solid #2563eb;
            padding: 1rem 1.2rem;
            border-radius: 6px;
            margin: 1rem 0;
            color: #1e3a8a;
        }

        .warning-box {
            background: #fff7ed;
            border-left: 4px solid #f97316;
            padding: 1rem 1.2rem;
            border-radius: 6px;
            margin: 1rem 0;
            color: #9a3412;
        }

        div[data-testid="stFileUploader"] {
            background-color: white;
            border-radius: 10px;
            padding: 0.5rem;
        }

        .stButton > button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
        }

        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. Load Environment Variables
# ============================================================

os.environ.pop("SSL_CERT_FILE", None)

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error(
        "OpenAI API key was not found. "
        "Please check the .env file."
    )
    st.stop()


# ============================================================
# 4. Initialize Models
# ============================================================

@st.cache_resource
def initialize_models():
    """Initialize the LLM and embedding model once per session."""

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    return llm, embeddings


llm, embeddings = initialize_models()


# ============================================================
# 5. Structured Output Schema
# ============================================================

class SkillEvidence(BaseModel):
    requirement: str = Field(
        description="Job Description requirement being evaluated"
    )

    status: str = Field(
        description="Evaluation status: Match or Missing"
    )

    evidence: str = Field(
        description="Short evidence from the resume supporting the status"
    )


class ResumeEvaluation(BaseModel):
    match_score: int = Field(
        description="Overall match score from 0 to 100"
    )

    matching_skills: List[str] = Field(
        description="Skills from the resume that match the Job Description"
    )

    missing_skills: List[str] = Field(
        description="Important Job Description requirements not found in the resume"
    )

    candidate_summary: str = Field(
        description="Short summary of the candidate's relevant experience"
    )

    strengths: List[str] = Field(
        description="Key strengths relevant to the Job Description"
    )

    weaknesses: List[str] = Field(
        description="Relevant gaps based only on the resume"
    )

    hiring_recommendation: str = Field(
        description="AI-assisted recommendation for recruiter review"
    )

    skill_evidence: List[SkillEvidence] = Field(
        description="Evidence-based assessment of individual Job Description requirements"
    )


output_parser = PydanticOutputParser(
    pydantic_object=ResumeEvaluation
)

format_instructions = output_parser.get_format_instructions()


# ============================================================
# 6. RAG Prompt
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are an AI Resume Screening Assistant.

Evaluate the candidate based only on the resume information
provided in the context.

Job Description:
{job_description}

Resume Context:
{context}

Evaluation rules:

1. Evaluate each important Job Description requirement individually.

2. If the resume contains explicit or clearly equivalent evidence,
   classify the requirement as "Match".

3. Only classify a requirement as "Missing" when there is no
   supporting evidence in the provided resume context.

4. Do not mark a requirement as missing simply because the exact
   wording is different.

5. Do not invent or assume experience that is not supported by
   the resume.

6. For related technologies, use reasonable technical equivalence
   only when the resume provides clear evidence.

7. For every requirement, provide a short evidence statement from
   the resume explaining the classification.

8. The match score should reflect the overall alignment between
   the candidate's documented experience and the Job Description.

9. The hiring recommendation is AI-assisted and must be treated
   as input for recruiter review, not as an automated final
   hiring decision.

10. Do not use protected or sensitive personal characteristics
    when evaluating the candidate.

Important technical equivalence examples:

- EKS or AKS can be treated as evidence of Kubernetes-based
  technology when clearly stated in the resume.

- Vector knowledge databases can be treated as evidence related
  to vector databases when clearly stated.

- Related technologies should not be treated as exact matches
  unless the resume provides reasonable supporting evidence.

Do not use outside knowledge about the candidate.

Provide:

- Matching skills
- Missing skills
- Candidate summary
- Strengths
- Weaknesses
- Match score from 0 to 100
- AI-assisted hiring recommendation for recruiter review
- Requirement-level skill evidence

The skill evidence must contain:

- requirement
- status: "Match" or "Missing"
- evidence

{format_instructions}
"""
)


# ============================================================
# 7. Helper Functions
# ============================================================

def load_resume_documents(uploaded_file):
    """
    Save an uploaded PDF temporarily and load it using
    LangChain's PyPDFLoader.
    """

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

    finally:
        Path(temp_path).unlink(missing_ok=True)

    return documents


def prepare_resume_chunks(documents, candidate_name):
    """
    Add candidate metadata and split resume documents
    into smaller chunks.
    """

    for document in documents:
        document.metadata["candidate"] = candidate_name

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


def create_vector_store(all_chunks):
    """
    Create a FAISS vector database from all resume chunks.
    """

    return FAISS.from_documents(
        documents=all_chunks,
        embedding=embeddings
    )


def retrieve_candidate_context(
    vector_store,
    candidate_name,
    queries,
    k=6
):
    """
    Retrieve relevant resume chunks for one candidate only.

    Multiple queries improve coverage across different
    Job Description requirements.
    """

    all_docs = []

    for query in queries:

        docs = vector_store.similarity_search(
            query,
            k=k,
            filter={"candidate": candidate_name}
        )

        all_docs.extend(docs)

    unique_docs = {}

    for document in all_docs:

        key = (
            document.metadata.get("candidate"),
            document.page_content
        )

        unique_docs[key] = document

    retrieved_docs = list(unique_docs.values())

    context = "\n\n".join(
        document.page_content
        for document in retrieved_docs
    )

    return retrieved_docs, context


def evaluate_resume(
    vector_store,
    candidate_name,
    job_description
):
    """
    Retrieve candidate-specific context and generate
    a structured resume evaluation.
    """

    retrieval_queries = [
        "Data Science Machine Learning AI experience years leadership teams",

        "Python Machine Learning Deep Learning NLP Generative AI LLM RAG",

        "AWS Azure GCP vector databases semantic search",

        "production AI ML Docker Kubernetes MLOps CI/CD deployment"
    ]

    retrieved_docs, context = retrieve_candidate_context(
        vector_store=vector_store,
        candidate_name=candidate_name,
        queries=retrieval_queries
    )

    formatted_prompt = prompt.format(
        job_description=job_description,
        context=context,
        format_instructions=format_instructions
    )

    response = llm.invoke(formatted_prompt)

    evaluation = output_parser.parse(
        response.content
    )

    evaluation.match_score = max(
        0,
        min(100, evaluation.match_score)
    )

    return evaluation, retrieved_docs


# ============================================================
# 8. Application Header
# ============================================================

st.markdown(
    """
    <div class="app-header">
        <h1>AI Resume Screening Assistant</h1>
        <p>
            AI-assisted resume evaluation using LangChain, RAG,
            FAISS and Generative AI
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 9. Sidebar
# ============================================================

with st.sidebar:

    st.markdown("## Screening Setup")

    st.markdown(
        """
        **Workflow**

        1. Enter or upload the Job Description
        2. Upload candidate resumes
        3. Run AI screening
        4. Review candidate evaluations
        5. Examine requirement-level evidence
        """
    )

    st.divider()

    st.markdown("### Technology")

    st.caption("LangChain")
    st.caption("OpenAI GPT-4o-mini")
    st.caption("OpenAI Embeddings")
    st.caption("FAISS Vector Database")
    st.caption("Retrieval-Augmented Generation")
    st.caption("Pydantic Structured Output")
    st.caption("Streamlit")

    st.divider()

    st.caption(
        "AI-generated results are intended to support "
        "recruiter review and should not be used as the "
        "sole basis for a hiring decision."
    )


# ============================================================
# 10. Job Description Input
# ============================================================

st.markdown(
    '<div class="section-title">1. Job Description</div>',
    unsafe_allow_html=True
)

jd_input_method = st.radio(
    "Choose how to provide the Job Description:",
    ["Paste Text", "Upload TXT File"],
    horizontal=True
)

job_description = ""

if jd_input_method == "Paste Text":

    job_description = st.text_area(
        "Job Description",
        height=260,
        placeholder=(
            "Paste the complete Job Description here..."
        ),
        help=(
            "Include responsibilities, required skills, "
            "experience and preferred qualifications."
        )
    )

else:

    jd_file = st.file_uploader(
        "Upload Job Description",
        type=["txt"],
        key="jd_upload"
    )

    if jd_file:

        job_description = jd_file.getvalue().decode(
            "utf-8",
            errors="ignore"
        )


if job_description.strip():

    st.success(
        f"Job Description loaded successfully "
        f"({len(job_description):,} characters)."
    )


# ============================================================
# 11. Resume Upload
# ============================================================

st.markdown(
    '<div class="section-title">2. Candidate Resumes</div>',
    unsafe_allow_html=True
)

uploaded_resumes = st.file_uploader(
    "Upload one or more candidate resumes",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload PDF resumes for evaluation."
)

if uploaded_resumes:

    st.info(
        f"{len(uploaded_resumes)} resume(s) selected."
    )

    resume_names = [
        Path(file.name).stem
        for file in uploaded_resumes
    ]

    st.write(
        "Selected candidates: "
        + ", ".join(resume_names)
    )


# ============================================================
# 12. Run Screening
# ============================================================

st.markdown(
    '<div class="section-title">3. Run AI Screening</div>',
    unsafe_allow_html=True
)

run_screening = st.button(
    "Run Resume Screening",
    type="primary",
    use_container_width=True
)


# ============================================================
# 13. Screening Pipeline
# ============================================================

if run_screening:

    if not job_description.strip():

        st.warning(
            "Please provide a Job Description before "
            "running the screening."
        )

        st.stop()

    if not uploaded_resumes:

        st.warning(
            "Please upload at least one resume before "
            "running the screening."
        )

        st.stop()

    all_chunks = []
    candidate_documents = {}

    progress_bar = st.progress(0)
    status_text = st.empty()

    # --------------------------------------------------------
    # Load and process resumes
    # --------------------------------------------------------

    for index, uploaded_file in enumerate(uploaded_resumes):

        candidate_name = Path(
            uploaded_file.name
        ).stem

        status_text.write(
            f"Processing resume: {candidate_name}"
        )

        documents = load_resume_documents(
            uploaded_file
        )

        chunks = prepare_resume_chunks(
            documents,
            candidate_name
        )

        candidate_documents[candidate_name] = documents
        all_chunks.extend(chunks)

        progress_bar.progress(
            (index + 1) / (len(uploaded_resumes) * 2)
        )

    # --------------------------------------------------------
    # Create FAISS vector database
    # --------------------------------------------------------

    status_text.write(
        "Creating FAISS vector database..."
    )

    vector_store = create_vector_store(
        all_chunks
    )

    progress_bar.progress(0.5)

    # --------------------------------------------------------
    # Evaluate candidates
    # --------------------------------------------------------

    evaluations = {}

    for index, uploaded_file in enumerate(
        uploaded_resumes
    ):

        candidate_name = Path(
            uploaded_file.name
        ).stem

        status_text.write(
            f"Evaluating candidate: {candidate_name}"
        )

        try:

            evaluation, retrieved_docs = evaluate_resume(
                vector_store=vector_store,
                candidate_name=candidate_name,
                job_description=job_description
            )

            evaluations[candidate_name] = {
                "evaluation": evaluation,
                "retrieved_docs": retrieved_docs
            }

        except Exception as error:

            st.error(
                f"Could not evaluate {candidate_name}: "
                f"{error}"
            )

        progress = 0.5 + (
            0.5 * (index + 1) / len(uploaded_resumes)
        )

        progress_bar.progress(progress)

    progress_bar.progress(1.0)
    status_text.success(
        "Resume screening completed successfully."
    )

    st.session_state["evaluations"] = evaluations


# ============================================================
# 14. Display Results
# ============================================================

if "evaluations" in st.session_state:

    evaluations = st.session_state["evaluations"]

    if not evaluations:

        st.error(
            "No candidate evaluations were generated."
        )

        st.stop()

    st.divider()

    st.markdown(
        '<div class="section-title">4. Screening Summary</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Summary metrics
    # --------------------------------------------------------

    total_candidates = len(evaluations)

    average_score = sum(
        item["evaluation"].match_score
        for item in evaluations.values()
    ) / total_candidates

    highest_score = max(
        item["evaluation"].match_score
        for item in evaluations.values()
    )

    total_missing = sum(
        len(item["evaluation"].missing_skills)
        for item in evaluations.values()
    )

    metric_columns = st.columns(4)

    with metric_columns[0]:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {total_candidates}
                </div>
                <div class="metric-label">
                    Candidates Evaluated
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric_columns[1]:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {average_score:.0f}
                </div>
                <div class="metric-label">
                    Average Match Score
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric_columns[2]:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {highest_score}
                </div>
                <div class="metric-label">
                    Highest Match Score
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric_columns[3]:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {total_missing}
                </div>
                <div class="metric-label">
                    Missing Requirements
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # Candidate comparison table
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Candidate Comparison</div>',
        unsafe_allow_html=True
    )

    comparison_rows = []

    for candidate_name, item in evaluations.items():

        evaluation = item["evaluation"]

        comparison_rows.append(
            {
                "Candidate": candidate_name,
                "Match Score": evaluation.match_score,
                "Matching Skills": len(
                    evaluation.matching_skills
                ),
                "Missing Skills": len(
                    evaluation.missing_skills
                ),
                "Evidence Items": len(
                    evaluation.skill_evidence
                )
            }
        )

    st.dataframe(
        comparison_rows,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # Individual candidate evaluations
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Candidate Evaluations</div>',
        unsafe_allow_html=True
    )

    for candidate_name, item in evaluations.items():

        evaluation = item["evaluation"]
        retrieved_docs = item["retrieved_docs"]

        with st.expander(
            f"{candidate_name}  •  Match Score: "
            f"{evaluation.match_score}/100",
            expanded=True
        ):

            # Candidate overview
            st.markdown("### Candidate Summary")

            st.write(
                evaluation.candidate_summary
            )

            # Strengths and weaknesses
            col1, col2 = st.columns(2)

            with col1:

                st.markdown("### Strengths")

                if evaluation.strengths:

                    for strength in evaluation.strengths:

                        st.markdown(
                            f"- {strength}"
                        )

                else:

                    st.write("No strengths identified.")


            with col2:

                st.markdown("### Areas to Review")

                if evaluation.weaknesses:

                    for weakness in evaluation.weaknesses:

                        st.markdown(
                            f"- {weakness}"
                        )

                else:

                    st.write(
                        "No significant gaps identified."
                    )


            # Matching skills
            st.markdown("### Matching Skills")

            if evaluation.matching_skills:

                st.write(
                    " • ".join(
                        evaluation.matching_skills
                    )
                )

            else:

                st.write(
                    "No matching skills identified."
                )


            # Missing skills
            st.markdown("### Missing Skills")

            if evaluation.missing_skills:

                for skill in evaluation.missing_skills:

                    st.markdown(
                        f"- {skill}"
                    )

            else:

                st.write(
                    "No important missing requirements identified."
                )


            # Recommendation
            st.markdown("### AI-Assisted Recommendation")

            st.info(
                evaluation.hiring_recommendation
            )


            # Requirement evidence
            st.markdown(
                "### Requirement-Level Evidence"
            )

            evidence_rows = []

            for evidence in evaluation.skill_evidence:

                evidence_rows.append(
                    {
                        "Requirement": evidence.requirement,
                        "Status": evidence.status,
                        "Evidence": evidence.evidence
                    }
                )

            if evidence_rows:

                st.dataframe(
                    evidence_rows,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.write(
                    "No requirement-level evidence available."
                )


            # Retrieved context
            with st.expander(
                "View Retrieved Resume Evidence"
            ):

                st.caption(
                    "The following text was retrieved from "
                    "the candidate's resume and supplied to "
                    "the RAG evaluation."
                )

                if retrieved_docs:

                    for index, document in enumerate(
                        retrieved_docs,
                        start=1
                    ):

                        st.markdown(
                            f"**Evidence Chunk {index}**"
                        )

                        st.write(
                            document.page_content
                        )

                        st.divider()

                else:

                    st.warning(
                        "No resume context was retrieved."
                    )


    # --------------------------------------------------------
    # Recruiter review notice
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        """
        <div class="warning-box">
            <strong>Recruiter Review Notice</strong><br><br>
            This application provides AI-assisted resume
            analysis based on the submitted Job Description
            and resume content. Results should be reviewed by
            a qualified recruiter or hiring professional.
            The system should not be used as the sole basis
            for a final employment decision.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 15. Footer
# ============================================================

st.markdown(
    """
    <div style="
        text-align: center;
        color: #6b7280;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
    ">
        AI Resume Screening Assistant |
        LangChain + RAG + FAISS + OpenAI + Streamlit | By Nakul Gupta
    </div>
    """,
    unsafe_allow_html=True
)