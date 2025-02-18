import streamlit as st
from paper_analyzer import PaperAnalyzer
from llm import LLMHandler
from vector_store import VectorStore
from citation_analyzer import CitationAnalyzer
from dotenv import load_dotenv
import plotly.graph_objects as go
import os


@st.cache_resource
def init_components():
    """Initialize components"""
    try:
        # Initialize core components
        llm_handler = LLMHandler()
        vector_store = VectorStore()
        paper_analyzer = PaperAnalyzer()
        citation_analyzer = CitationAnalyzer()

        return {
            "paper_analyzer": paper_analyzer,
            "llm_handler": llm_handler,
            "vector_store": vector_store,
            "citation_analyzer": citation_analyzer
        }
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        return None


def main():
    # Load environment variables
    load_dotenv()

    # Page config
    st.set_page_config(page_title="Paper Summarizer", page_icon="📚", layout="wide")
    st.title("📚 Academic Paper Summarizer")

    # Initialize components
    if "components" not in st.session_state:
        components = init_components()
        if components is None:
            st.error("Failed to initialize application components")
            return
        st.session_state.components = components

    # Verify components are available
    required_components = ["paper_analyzer", "llm_handler", "vector_store", "citation_analyzer"]
    for component in required_components:
        if component not in st.session_state.components:
            st.error(f"Missing required component: {component}")
            return

    # Search interface
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("🔍 Enter your research topic:")
    with col2:
        max_results = st.slider("Number of papers", 1, 10, 5)

    if st.button("🚀 Search Papers"):
        with st.spinner("Searching papers..."):
            try:
                papers = st.session_state.components["paper_analyzer"].search_papers(
                    search_query, max_results=max_results
                )
                st.session_state.papers = papers
                st.success(f"Found {len(papers)} papers!")
            except Exception as e:
                st.error(f"Search failed: {str(e)}")

    # Display papers
    if "papers" in st.session_state:
        for i, paper in enumerate(st.session_state.papers):
            display_paper_card(paper, i)


def display_paper_card(paper: dict, index: int):
    """Display a paper with enhanced UI and analysis"""
    with st.expander(f"📄 {index + 1}. {paper['title']}", expanded=index == 0):
        # Basic paper info
        st.write(f"**Authors:** {', '.join(paper['authors'])}")
        st.write(f"**Published:** {paper['published']}")
        st.write(f"**PDF:** [{paper['pdf_url']}]({paper['pdf_url']})")

        # Create tabs for different analyses
        tab1, tab2 = st.tabs(["📝 Summary & Analysis", "🔍 Keywords & Categories"])

        # Summary Tab
        with tab1:
            summary_type = st.radio(
                "Summary Type", ["Basic", "Enhanced"], key=f"type_{paper['title']}"
            )

            if st.button("Generate Summary", key=f"sum_{paper['title']}"):
                with st.spinner("Generating summary..."):
                    try:
                        if summary_type == "Basic":
                            summary = st.session_state.components["llm_handler"].generate_summary(
                                paper["summary"]
                            )
                            st.write("### Summary")
                            st.write(summary)

                            # Store in vector DB
                            stored = st.session_state.components["vector_store"].add_paper_with_summary(
                                paper, summary
                            )
                            if stored:
                                st.success("Summary stored for future reference")

                        else:  # Enhanced summary
                            similar_papers = st.session_state.components["vector_store"].get_similar_papers(
                                paper["summary"]
                            )

                            if not similar_papers:
                                st.warning(
                                    "No similar papers found in database. Generating basic summary..."
                                )
                                summary = st.session_state.components["llm_handler"].generate_summary(
                                    paper["summary"]
                                )
                            else:
                                summary = st.session_state.components["llm_handler"].generate_refined_summary(
                                    paper["summary"], similar_papers
                                )

                            st.write("### Enhanced Summary")
                            st.write(summary)

                            # Show related papers
                            if similar_papers:
                                st.write("### Related Research")
                                for sim_paper in similar_papers:
                                    st.write(f"- **{sim_paper['title']}**")
                                    st.write(
                                        f"  *{sim_paper['generated_summary'][:200]}...*"
                                    )

                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")

        # Keywords & Categories Tab
        with tab2:
            if st.button("Extract Keywords", key=f"key_{paper['title']}"):
                with st.spinner("Analyzing paper..."):
                    try:
                        # Extract keywords and categories using LLM
                        analysis_prompt = f"""
                        Analyze this paper and provide:
                        1. Key technical terms and concepts (max 5)
                        2. Main research areas/categories
                        3. Target audience
                        4. Technical complexity level (Beginner/Intermediate/Advanced)
                        5. Practical applications

                        Paper Title: {paper['title']}
                        Abstract: {paper['summary']}
                        """
                        
                        analysis = st.session_state.components["llm_handler"].generate_summary(
                            analysis_prompt
                        )
                        
                        st.write("### 📊 Paper Analysis")
                        st.write(analysis)
                        
                        # Show paper stats
                        st.write("### 📈 Paper Statistics")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Authors", len(paper['authors']))
                            if 'categories' in paper:
                                st.metric("Categories", len(paper['categories']))
                                
                        with col2:
                            # Calculate summary length in words
                            word_count = len(paper['summary'].split())
                            st.metric("Abstract Length", f"{word_count} words")
                            
                            # Calculate estimated reading time
                            reading_time = word_count // 200  # Assuming 200 words per minute
                            st.metric("Est. Reading Time", f"{reading_time} min")

                    except Exception as e:
                        st.error(f"Error analyzing paper: {str(e)}")


if __name__ == "__main__":
    main()
