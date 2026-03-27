import streamlit as st

from cv_pipeline import build_vectorstore
from retrieval import build_retriever, run_rag


# Page setup
st.set_page_config(page_title="📄 HR CV System")
st.title("Chat with CVs")


# Sidebar — CV Upload
with st.sidebar:
    st.header("📂 CV Upload Panel")

    uploaded_files = st.file_uploader(
        "Upload exactly 5 CV PDFs",
        type="pdf",
        accept_multiple_files=True,
    )
    if uploaded_files:
        if len(uploaded_files) != 5:
            st.error("❌ You must upload exactly 5 CVs.")
            st.stop()

        st.success("✅ 5 CVs uploaded successfully.")



# In the first run, there are no CVs loaded, so we set a flag in session state to track that.
if "cv_loaded" not in st.session_state:
    st.session_state.cv_loaded = False

# When the uploaded CVs are not processed yet, we set a flag to trigger the vectorstore build.
if not st.session_state.cv_loaded:
    st.session_state.rebuild_collection = True

# Creates an empty list to store all chat messages
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



# Build / Rebuild Vectorstore
if uploaded_files and len(uploaded_files) == 5:

    # Build vectorstore only the first time CVs are uploaded
    if st.session_state.get("rebuild_collection", True):

        # vectorstore, total_chunks = build_vectorstore(uploaded_files)
        vectorstore, bm25_index, chunks, total_chunks = build_vectorstore(uploaded_files)

        st.session_state.total_chunks = total_chunks
        st.session_state.vectorstore = vectorstore
        st.session_state.bm25_index = bm25_index
        st.session_state.chunks = chunks
        st.session_state.cv_loaded = True
        st.session_state.rebuild_collection = False

    # Use the existing vectorstore in session state for subsequent runs until session ends
    else:
        vectorstore = st.session_state.vectorstore # 



# Sidebar — Metrics
with st.sidebar:
    st.markdown("### 📊 System Stats")

    total_cvs = len(uploaded_files) if uploaded_files else 0
    st.metric("Total CVs", total_cvs)
    st.metric("Total Chunks", st.session_state.get("total_chunks", 0))


# Build the retriever only once after the vectorstore is built, and store it in session state for reuse across runs
if "vectorstore" in st.session_state:
    if "retriever" not in st.session_state:
        st.session_state.retriever = build_retriever(st.session_state.vectorstore)
    multi_query_retriever = st.session_state.retriever



# Render Chat History
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):

        if chat["role"] == "user":
            st.markdown(chat["message"])

        elif chat["role"] == "assistant":

            retrieved_docs = chat.get("retrieved_docs", [])
            query_list = chat.get("multi_queries", [])
            history_groups = chat.get("candidate_groups", {})
            history_years  = chat.get("years_map", {})

            # Only show metadata if there are actual results
            if retrieved_docs:
                st.caption("🧠 CVs Analyzed")
                displayed_chunk_count = sum(len(texts) for texts in history_groups.values()) if history_groups else len(retrieved_docs)
                st.caption(f"🔎 Retrieved {displayed_chunk_count} relevant chunks")

                if query_list:
                    with st.expander("🧠 Generated Multi-Queries"):
                        for q in query_list:
                            st.write(q)

            # Answer the query
            st.markdown(chat["message"])

            # Only show metadata if there are actual results
            if retrieved_docs:

                with st.expander("📚 Retrieved Chunks Used"):
                    if history_groups:
                        for candidate, texts in history_groups.items():
                            years = history_years.get(candidate, "N/A")
                            st.markdown(f"### 👤 {candidate} — {years} yrs experience")
                            for j, text in enumerate(texts, 1):
                                st.markdown(f"**Chunk {j}:**")
                                st.write(text)
                                st.write("------")
                    else:
                        for i, doc in enumerate(retrieved_docs):
                            st.write(f"Chunk {i+1} | File: {doc.metadata.get('file_name')}")
                            if doc.metadata.get("years_of_experience"):
                                st.write(f"Years of Experience: {doc.metadata.get('years_of_experience')}")
                            st.write(doc.page_content)
                            st.write("------")



# Chat Input & Response Generation
query = st.chat_input("Ask a question about candidates...")

if query and "vectorstore" in st.session_state:

    # Save and render user message
    st.session_state.chat_history.append({"role": "user", "message": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Assistant response
    with st.chat_message("assistant"):

        with st.spinner("Analyzing CVs..."):

            if query and "retriever" in st.session_state:
        
                response, retrieved_docs, query_list, candidate_groups, years_map = run_rag(
                    query,
                    st.session_state.retriever,
                    st.session_state.bm25_index,
                    st.session_state.chunks
                )

        # Only show metadata if there are actual results
        if retrieved_docs:
            st.caption("🧠 CVs Analyzed")
            displayed_chunk_count = sum(len(texts) for texts in candidate_groups.values()) if candidate_groups else len(retrieved_docs)
            st.caption(f"🔎 Retrieved {displayed_chunk_count} relevant chunks")

            if query_list:
                with st.expander("🧠 Generated Multi-Queries"):
                    for q in query_list:
                        st.write(q)

        # Answer the query
        st.markdown(response)

        # Only show metadata if there are actual results
        if retrieved_docs:

            with st.expander("📚 Retrieved Chunks Used"):
                if candidate_groups:
                    for candidate, texts in candidate_groups.items():
                        years = years_map.get(candidate, "N/A")
                        st.markdown(f"### 👤 {candidate} — {years} yrs experience")
                        for j, text in enumerate(texts, 1):
                            st.markdown(f"**Chunk {j}:**")
                            st.write(text)
                            st.write("------")
                            
                else:
                    # Fallback for experience queries (no candidate_groups)
                    for i, doc in enumerate(retrieved_docs):
                        st.write(f"Chunk {i+1} | File: {doc.metadata.get('file_name')}")
                        st.write(f"Years of Experience: {doc.metadata.get('years_of_experience')}")
                        st.write(doc.page_content)
                        st.write("------")

    # Save assistant message to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "message": response,
        "retrieved_docs": retrieved_docs,
        "retrieval_count": len(retrieved_docs),
        "multi_queries": query_list,
        "candidate_groups": candidate_groups,
        "years_map": years_map,
    })