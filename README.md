# AI Resume Screening Assistant

An AI-powered Resume Screening Assistant built using **LangChain, Retrieval-Augmented Generation (RAG), OpenAI GPT-4o-mini, FAISS, Pydantic, and Streamlit**.

The application helps recruiters evaluate multiple resumes against a Job Description (JD) and generates structured, evidence-based candidate evaluations.

---

## 1. Project Overview

Recruiters often need to review multiple resumes against the same Job Description. Manually comparing skills, experience, technologies, and requirements can be time-consuming.

This project provides an AI-assisted solution that retrieves relevant information from each candidate's resume and evaluates it against a given Job Description.

The system generates:

* Match Score (0–100)
* Matching Skills
* Missing Skills
* Candidate Summary
* Strengths
* Weaknesses
* Requirement-Level Evidence
* AI-Assisted Hiring Recommendation

The application is designed to **support recruiter review**, rather than make an automated final hiring decision.

---

## 2. Project Objective

The objective of this project is to build a practical **LangChain-based RAG application** that can:

1. Accept multiple PDF resumes.
2. Process and split resume content into smaller chunks.
3. Convert resume chunks into vector embeddings.
4. Store embeddings in a FAISS vector database.
5. Retrieve candidate-specific information relevant to the Job Description.
6. Use an LLM to evaluate the retrieved resume information.
7. Generate structured evaluation results using Pydantic.
8. Present the results through a professional Streamlit interface.

---

## 3. Key Features

### Resume Processing

* Upload one or more PDF resumes.
* Extract resume text using LangChain's PDF loader.
* Split documents into overlapping text chunks.

### RAG-Based Evaluation

* Generate embeddings using OpenAI's embedding model.
* Store embeddings in a FAISS vector database.
* Perform candidate-specific semantic retrieval.
* Use retrieved resume evidence as context for the LLM.

### Structured AI Evaluation

The system generates:

* Match Score
* Matching Skills
* Missing Skills
* Candidate Summary
* Strengths
* Weaknesses
* AI-Assisted Hiring Recommendation
* Requirement-Level Skill Evidence

### Streamlit Application

The application provides a recruiter-friendly interface where users can:

* Enter or upload a Job Description.
* Upload multiple candidate resumes.
* Run AI-based resume screening.
* Compare candidate evaluation results.
* Review requirement-level evidence.
* Inspect the retrieved resume evidence used by the RAG pipeline.

---

## 4. System Architecture

```text
                  Job Description
                         │
                         ▼
                  ┌──────────────┐
                  │   Streamlit  │
                  │  Application │
                  └──────┬───────┘
                         │
                         │
              ┌──────────▼──────────┐
              │    Resume PDFs      │
              └──────────┬──────────┘
                         │
                         ▼
                PDF Document Loader
                         │
                         ▼
                  Text Chunking
                         │
                         ▼
                OpenAI Embeddings
                         │
                         ▼
                 FAISS Vector DB
                         │
                         ▼
             Candidate-Specific Retrieval
                         │
                         ▼
                Relevant Resume Context
                         │
                         ▼
                  LangChain RAG
                         │
                         ▼
                    GPT-4o-mini
                         │
                         ▼
                Pydantic Output Parser
                         │
                         ▼
             Structured Candidate Evaluation
                         │
                         ▼
                Recruiter Review
```

---

## 5. Technologies Used

| Technology                     | Purpose                                           |
| ------------------------------ | ------------------------------------------------- |
| Python                         | Application development                           |
| LangChain                      | RAG pipeline and LLM orchestration                |
| OpenAI GPT-4o-mini             | Resume evaluation and natural language generation |
| OpenAI Embeddings              | Convert resume text into vector representations   |
| FAISS                          | Vector database and semantic retrieval            |
| PyPDF                          | PDF document extraction                           |
| RecursiveCharacterTextSplitter | Resume text chunking                              |
| Pydantic                       | Structured output validation                      |
| Streamlit                      | Web application interface                         |
| python-dotenv                  | Environment variable management                   |

---

## 6. Project Structure

```text
ai_resume_screening_assistant/
│
├── app.py
├── code.ipynb
├── requirements.txt
├── README.md
├── .gitignore
│
└── data/
    ├── resumes/
    │   ├── Eric Smith.pdf
    │   ├── Hoon Y..pdf
    │   └── James Howard.pdf
    │
    └── job_descriptions/
        └── senior_data_scientist_genai_engineer.txt
```

### File Description

**`code.ipynb`**

Contains the development and implementation of the complete RAG pipeline, including document loading, chunking, embeddings, FAISS vector database creation, retrieval, structured output, candidate evaluation, comparison, and validation.

**`app.py`**

Contains the Streamlit application that provides the recruiter-facing interface.

**`requirements.txt`**

Contains the Python dependencies required to run the project.

**`data/resumes/`**

Contains sample PDF resumes used for testing.

**`data/job_descriptions/`**

Contains the sample Job Description used for testing the application.

**`.env`**

Stores the OpenAI API key locally. This file is intentionally excluded from GitHub using `.gitignore`.

---

## 7. RAG Workflow

The project follows a Retrieval-Augmented Generation workflow.

### Step 1 — Resume Loading

PDF resumes are loaded using LangChain's `PyPDFLoader`.

### Step 2 — Text Chunking

Resume content is divided into smaller overlapping chunks using `RecursiveCharacterTextSplitter`.

### Step 3 — Embeddings

Each chunk is converted into a numerical vector using OpenAI's `text-embedding-3-small` model.

### Step 4 — Vector Database

The embeddings are stored in a FAISS vector database.

### Step 5 — Candidate-Specific Retrieval

Relevant resume chunks are retrieved for each candidate using multiple retrieval queries.

This helps prevent information from different candidates from being mixed during evaluation.

### Step 6 — RAG Prompt

The retrieved resume context and Job Description are provided to GPT-4o-mini through a LangChain prompt.

The model is instructed to evaluate the candidate using only the provided resume evidence.

### Step 7 — Structured Output

The response is parsed using a Pydantic output schema.

This provides a consistent structure for the candidate evaluation.

### Step 8 — Recruiter Review

The structured results are presented through Streamlit for recruiter review.

---

## 8. How to Set Up the Project

### Step 1 — Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate to the project folder:

```bash
cd ai_resume_screening_assistant
```

### Step 2 — Create the Virtual Environment

Create and activate the project environment as required for your local setup.

For example:

```bash
python -m venv myenv
```

Activate it on Windows:

```bash
myenv\Scripts\activate
```

### Step 3 — Install Dependencies

Install the required packages using:

```bash
pip install -r requirements.txt
```

### Step 4 — Configure the OpenAI API Key

Create a `.env` file in the project root:

```text
OPENAI_API_KEY=your_openai_api_key
```

Do not commit the `.env` file to GitHub.

---

## 9. Run the Streamlit Application

From the project directory, run:

```bash
streamlit run app.py
```

The application will open in the browser.

---

## 10. Using the Application

### Step 1 — Provide the Job Description

The application allows the recruiter to either:

* Paste the Job Description directly into the application, or
* Upload a `.txt` Job Description file.

### Step 2 — Upload Resumes

Upload one or more candidate resumes in PDF format.

### Step 3 — Run Screening

Click:

**Run Resume Screening**

The system processes the resumes, retrieves relevant information, and generates structured candidate evaluations.

### Step 4 — Review Results

The application displays:

* Candidate Match Score
* Matching Skills
* Missing Skills
* Candidate Summary
* Strengths
* Areas to Review
* AI-Assisted Recommendation
* Requirement-Level Evidence
* Retrieved Resume Evidence

---

## 11. Example Use Cases

The application can be used for scenarios such as:

### Use Case 1 — Individual Resume Evaluation

Evaluate a candidate's resume against a specific Job Description.

### Use Case 2 — Multiple Candidate Evaluation

Upload multiple resumes and evaluate each candidate against the same Job Description.

### Use Case 3 — Skill Gap Identification

Identify important Job Description requirements that are not supported by the candidate's resume.

### Use Case 4 — Evidence-Based Screening

Review the resume evidence retrieved by the RAG pipeline for each requirement.

### Use Case 5 — Candidate Comparison

Compare structured evaluation results across multiple candidates.

---

## 12. Sample Test Data

The project was tested using three sample resumes:

* Eric Smith
* Hoon Y.
* James Howard

The application was tested by evaluating all three resumes against the sample Senior Data Scientist / Generative AI Engineer Job Description.

The complete pipeline and Streamlit application were successfully tested without application errors.

---

## 13. Validation

The project includes validation steps to check the generated structured evaluations.

The validation process checks:

* Match score validity
* Candidate summary availability
* Requirement-level evidence
* Valid Match/Missing statuses
* Availability of retrieved resume context

These checks help identify incomplete or unreliable structured responses before displaying results to the recruiter.

---

## 14. Limitations

This application has several limitations:

* The quality of the evaluation depends on the quality and completeness of the submitted resume.
* LLM-generated evaluations can vary between runs.
* Semantic retrieval may occasionally miss relevant resume information.
* Match scores should not be interpreted as objective hiring scores.
* The system should not be used as the sole basis for an employment decision.
* Recruiters should review the underlying resume evidence before making a final decision.

---

## 15. Responsible AI and Human-in-the-Loop

The application is designed as an **AI-assisted screening tool**.

The generated recommendation is intended to support recruiter review and should not replace human judgment.

The system should focus on job-relevant qualifications and documented experience and should not be used to infer or evaluate protected or sensitive personal characteristics.

---

## 16. Future Enhancements

Potential future improvements include:

* Support for additional document formats such as DOCX.
* Persistent vector database storage.
* Improved retrieval and reranking.
* More detailed evaluation criteria.
* Resume and Job Description analytics dashboards.
* Authentication and role-based access.
* Automated evaluation benchmarking.
* Cloud deployment.
* Integration with an Applicant Tracking System (ATS).

---

## 17. Conclusion

This project demonstrates how **LangChain, Retrieval-Augmented Generation, vector search, structured LLM output, and Streamlit** can be combined to build a practical AI application for resume screening.

The system focuses on retrieving relevant candidate information from resumes and providing structured, evidence-based evaluations that can assist recruiters during the screening process.

---

## Author

**Nakul Gupta**

Data Science | Generative AI | Machine Learning | HR Technology
