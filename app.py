import streamlit as st
import re
from pipeline import run_research_pipeline  

# ── PAGE CONFIGURATION ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Case Companion",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CUSTOM CSS FOR BLACK BACKGROUND ───────────────────────────────────────────
st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    
    /* Ensuring text inputs and tabs look good on dark backgrounds */
    .stTextInput > label, .stTabs > div > div > button > div > p {
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# ── APPLICATION HEADER ────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align: center; font-size: 4.5rem; padding-bottom: 20px;'>Case Companion</h1>", 
    unsafe_allow_html=True
)
st.write("---")

# Initialize session state for caching results and UI flow
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = None
if "topic" not in st.session_state:
    st.session_state.topic = ""

# Helper function to parse Groq 429 rate limit errors for time
def extract_wait_time(error_message):
    match = re.search(r"try again in (?:(\d+)m)?(?:([\d.]+)s)?", str(error_message))
    if match:
        mins = match.group(1)
        secs = match.group(2)
        if mins and secs:
            return f"{mins} min {float(secs):.0f} sec"
        elif mins:
            return f"{mins} min"
        elif secs:
            return f"{float(secs):.0f} sec"
    return "a few minutes"

# ── CENTER SEARCH BAR (VANISHES AFTER SUBMISSION) ─────────────────────────────
if st.session_state.pipeline_results is None:
    # Use empty container so we can clear it from the UI later
    input_container = st.empty()
    
    with input_container.container():
        # Using columns to center the input bar
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            topic = st.text_input(
                "Research Topic / Market Segment:", 
                placeholder="e.g., Tier 1 protein market in India for Zepty",
                label_visibility="collapsed"
            )
            run_button = st.button("Run Research Pipeline", type="primary", use_container_width=True)

    if run_button:
        if not topic.strip():
            st.error("Please enter a valid topic before running.")
        else:
            st.session_state.topic = topic
            input_container.empty() # Makes the search bar vanish
            
            with st.spinner("Wait while we are pulling out that 23 verstappen performance..."):
                try:
                    # Run the backend pipeline
                    results = run_research_pipeline(topic)
                    
                    # Save to session state and rerun to show results
                    st.session_state.pipeline_results = results
                    st.rerun()
                    
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "rate_limit" in error_str.lower():
                        wait_time = extract_wait_time(error_str)
                        st.error(f"Limit reached please try again after {wait_time}")
                    else:
                        st.error(f"An unexpected error occurred: {e}")
                    
                    # Reset state so the search bar comes back
                    st.session_state.pipeline_results = None

# ── RENDERING THE OUTPUTS ─────────────────────────────────────────────────────
if st.session_state.pipeline_results:
    results = st.session_state.pipeline_results
    
    # Header showing what was analyzed and a reset button
    colA, colB = st.columns([4, 1])
    with colA:
        st.subheader(f"Analysis for: {st.session_state.topic}")
    with colB:
        if st.button("New Search", use_container_width=True):
            st.session_state.pipeline_results = None
            st.session_state.topic = ""
            st.rerun()

    # Structure the UI layout to only show Report and Sources
    tab1, tab2 = st.tabs([
        "Strategic Report Draft", 
        "Sources & Context"
    ])
    
    with tab1:
        st.subheader("Generated Report")
        if results.get("report"):
            st.markdown(results["report"])
        else:
            st.info("The synthesis step completed but returned an empty report frame.")
            
    with tab2:
        st.subheader("Extracted Grounding Data Context")
        st.caption("Review the truncated raw text injected into downstream context windows.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Web Search Indexing")
            st.text_area(
                "Cleaned Search Snippets", 
                value=results.get("search_results", "No search data found."), 
                height=400,
                label_visibility="collapsed"
            )
        with col2:
            st.markdown("### Target Page Scraping")
            st.text_area(
                "Cleaned Scraped Content", 
                value=results.get("scrapped_content", "No deep scrape data compiled."), 
                height=400,
                label_visibility="collapsed"
            )
            