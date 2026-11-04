from pathlib import Path
from typing import TypedDict

import ollama
import streamlit as st
from langgraph.graph import END, START, StateGraph
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Set up the page shown in the browser
st.set_page_config(
    page_title="Community Health Guideline Navigator",
    page_icon="🩺",
    layout="wide",
)

# Find the project folder and the guideline PDFs
PROJECT_FOLDER = Path(__file__).resolve().parent
GUIDELINE_FOLDER = PROJECT_FOLDER / "data" / "guidelines"
MODEL_NAME = "llama3.2:latest"


# Describe the information passed between workflow steps
class AgentState(TypedDict):
    cadre: str
    question: str
    evidence: list[dict]
    answer: str


# Load the guidelines and prepare the local search index
@st.cache_resource
def load_resources():
    guideline_pages = []

    # Read each PDF and keep its cadre, filename, page, and text together
    for pdf_path in sorted(GUIDELINE_FOLDER.glob("*.pdf")):
        if pdf_path.name.startswith("CHEW_"):
            cadre = "CHEW"
        elif pdf_path.name.startswith("CHO_"):
            cadre = "CHO"
        else:
            continue

        reader = PdfReader(str(pdf_path))

        for pdf_page, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = " ".join(text.split())

            if text:
                guideline_pages.append({
                    "cadre": cadre,
                    "source": pdf_path.name,
                    "pdf_page": pdf_page,
                    "text": text,
                })

    # Stop with a clear message if the PDFs are missing or unreadable
    if not guideline_pages:
        raise FileNotFoundError(
            "No readable CHEW or CHO PDFs found in data/guidelines."
        )

    # Build a local word-search index from all extracted pages
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )
    page_matrix = vectorizer.fit_transform(
        [page["text"] for page in guideline_pages]
    )

    # Search the selected cadre's pages and return the best matches
    def search_guidance(query, cadre, top_k=3):
        query_vector = vectorizer.transform([query])
        scores = cosine_similarity(query_vector, page_matrix).flatten()

        matching_pages = [
            index
            for index, page in enumerate(guideline_pages)
            if page["cadre"] == cadre
            and "table of contents" not in page["text"].lower()
        ]

        ranked_pages = sorted(
            matching_pages,
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []
        for index in ranked_pages[:top_k]:
            if scores[index] > 0:
                results.append(guideline_pages[index])

        return results

    # Build the LangGraph workflow around the local search
    def retrieve_node(state: AgentState):
        evidence = search_guidance(
            state["question"],
            state["cadre"],
        )
        return {"evidence": evidence}

    # Choose whether to draft an answer or explain that no source was found
    def route_after_retrieval(state: AgentState):
        if state["evidence"]:
            return "draft"
        return "no_guidance"

    # Ask the local model to summarise only the pages found by search
    def draft_node(state: AgentState):
        source_text = "\n\n".join(
            f'Source: {item["source"]}, PDF page {item["pdf_page"]}\n'
            f'{item["text"]}'
            for item in state["evidence"]
        )

        system_message = (
            "You are a guideline navigation assistant for trained CHEWs and CHOs. "
            "Use only the supplied Standing Orders text. Do not add medical facts, "
            "diagnoses, medicines, doses, or instructions that are not in the text. "
            "If the text does not answer the question, say so. Do not provide filenames or PDF page numbers. The app will display retrieved source pages separately. "
            " Your answer is a draft for health-worker "
            "review, not an independent clinical decision."
        )

        user_message = (
            f'Cadre: {state["cadre"]}\n'
            f'Question: {state["question"]}\n\n'
            f"Retrieved Standing Orders text:\n{source_text}"
        )

        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            options={"temperature": 0},
        )

        return {"answer": response.message.content}

    # Do not ask the model to answer when search finds no source text
    def no_guidance_node(state: AgentState):
        return {
            "answer": (
                "I could not find relevant text in this cadre's Standing Orders. "
                "No summary was generated. Please check the manual or ask a supervisor."
            )
        }

    # Connect the workflow steps
    builder = StateGraph(AgentState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("draft", draft_node)
    builder.add_node("no_guidance", no_guidance_node)
    builder.add_edge(START, "retrieve")
    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "draft": "draft",
            "no_guidance": "no_guidance",
        },
    )
    builder.add_edge("draft", END)
    builder.add_edge("no_guidance", END)

    return guideline_pages, builder.compile()


# Load the manuals and build the workflow once when the app starts
try:
    guideline_pages, health_agent = load_resources()
except Exception as error:
    st.error(f"Could not load the guidelines or build the agent: {error}")
    st.stop()


# Show the app title and explain its purpose
st.title("Community Health Guideline Navigator")
st.write("A local prototype for CHEWs and CHOs using Nigeria's Standing Orders.")
st.warning(
    "Demonstration prototype only. Use fictional cases, enter no patient identifiers, "
    "and check the cited pages in the official manual."
)

# Let the health worker choose which manual to search
cadre = st.selectbox("Select your cadre", ["CHEW", "CHO"])

# Let the health worker ask a question about the selected manual
question = st.text_area(
    "What topic or question should the agent search for?",
    placeholder="For example: Find the section on newborn emergencies.",
)

# Ask the user to confirm safe demo use
review_acknowledged = st.checkbox(
    "I will use fictional examples and review the cited manual pages."
)

# Run the graph when the user presses the button
if st.button("Search the Standing Orders", type="primary"):
    if not question.strip():
        st.warning("Enter a topic or question first.")
    elif not review_acknowledged:
        st.warning("Please confirm that you will use fictional examples and review sources.")
    else:
        try:
            with st.spinner("Searching the selected manual and asking Ollama..."):
                result = health_agent.invoke({
                    "cadre": cadre,
                    "question": question,
                    "evidence": [],
                    "answer": "",
                })

            st.session_state["answer"] = result["answer"]
            st.session_state["evidence"] = result["evidence"]

        except Exception as error:
            st.error(
                f"The search could not be completed: {error}. "
                "Check that Ollama is running and llama3.2:latest is installed."
            )

# Display the draft answer and the pages it used
if st.session_state.get("answer"):
    st.subheader("AI-generated draft — health-worker review required")
    st.write(st.session_state["answer"])

    st.markdown("**Search-matched source pages:**")
    for item in st.session_state.get("evidence", []):
        st.markdown(
            f'- `{item["source"]}` — PDF page {item["pdf_page"]}'
        )

    with st.expander("View the retrieved source pages"):
        for item in st.session_state.get("evidence", []):
            st.markdown(
                f'**{item["source"]} — PDF page {item["pdf_page"]}**'
            )
            st.write(item["text"])
