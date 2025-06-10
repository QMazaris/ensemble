 Phase 1: Audit and Instrument the Code
🔍 1. Identify Expensive Calls
Add logging or print statements to track:

API calls

Graph generation

Data processing (e.g., metrics, confusion matrices)

Note which functions rerun every time any interaction occurs.

📝 2. Document State Dependencies
List which data or graphs depend on:

Backend API calls

User inputs

Pipeline outputs

Create a flowchart or markdown note of state triggers.

⚙️ Phase 2: Introduce Lazy Loading + Caching
💾 3. Implement st.session_state Guarding for Each Tab
In app.py or tabs/*.py:

python
Copy
Edit
if "model_analysis_loaded" not in st.session_state:
    if st.button("Load Model Analysis"):
        st.session_state.model_analysis_data = fetch_metrics()
        st.session_state.model_analysis_loaded = True
Then render graphs:

python
Copy
Edit
if "model_analysis_loaded" in st.session_state:
    render_model_graphs(st.session_state.model_analysis_data)
📦 4. Cache Expensive Backend/Data Functions
Convert any expensive function to @st.cache_data:

python
Copy
Edit
@st.cache_data(ttl=600)
def fetch_metrics():
    return requests.get("/results/metrics").json()
Also cache derived data:

python
Copy
Edit
@st.cache_data
def compute_confusion_matrix(metrics_data):
    ...
📉 Phase 3: Optimize Graph Rendering
🎨 5. Memoize Figures
Cache figure objects if data hasn’t changed:

python
Copy
Edit
if "cached_fig" not in st.session_state:
    st.session_state.cached_fig = generate_plot(metrics_data)

st.plotly_chart(st.session_state.cached_fig)
Or wrap in cached function:

python
Copy
Edit
@st.cache_data
def generate_plot(data):
    fig = ...
    return fig
📊 Phase 4: Conditional UI Rendering
🧩 6. Load Tab Contents Lazily
Only load tab contents when clicked, with guards:

python
Copy
Edit
if selected_tab == "Downloads":
    if "downloads_loaded" not in st.session_state:
        if st.button("Load Downloads"):
            st.session_state.download_data = get_downloads()
            st.session_state.downloads_loaded = True
    else:
        render_download_tab(st.session_state.download_data)
🧼 Phase 5: Final Polish & UX Improvements
🔃 7. Add Manual Refresh Buttons
Instead of auto-rerun on every change, add:

python
Copy
Edit
if st.button("Refresh Metrics"):
    st.session_state.model_analysis_data = fetch_metrics()
    st.session_state.model_analysis_loaded = True
🧠 8. Show Spinner During Loading
Use:

python
Copy
Edit
with st.spinner("Loading..."):
    ...
🚦 9. Set Time-To-Live (TTL) for Cached Data
Use @st.cache_data(ttl=600) to expire stale data after 10 minutes.

🧪 Phase 6: Testing and Monitoring
✅ 10. Test All Tabs Individually
Ensure each loads only when requested.

Ensure refresh buttons work.

Confirm no redundant API calls or rerenders.

📈 11. Profile Performance
Use logging or tools like:

time.time() to log timing

streamlit statics or browser dev tools to monitor load times

🧩 Optional: Advanced Refactors
🔧 Modular Refactor (Optional, Later)
Break each tab into:

load_data()

render_view()

refresh_handler()

Keep app.py clean and declarative


In overview, made it only call the api when the pipeline has run or the page is reloaded. Would prop need to do that for the rest to get ideal results. 

In the future, I should make the api calls more specific / useful so the front end needs to do less processing. Also, there should be one centeralized function that properly handles all the cashing so there is no garbage where only some of it works. 