
import json
import time
import streamlit as st

try:
    from agent_logic import crawl_web, DEFAULT_UA
except Exception as e:
    st.error("Failed to import agent_logic. See details below.")
    st.exception(e)
    st.stop()

st.set_page_config(page_title="Agentic Web Crawler", page_icon="🕷️", layout="wide")

st.title("🕷️ Agentic Web Crawler (LangChain)")
st.caption("Enter a URL. The agent fetches the page, extracts text/metadata/links, and optionally summarizes it with an LLM.")

with st.sidebar:
    st.header("Settings")
    url = st.text_input("URL", placeholder="https://example.com")
    col1, col2 = st.columns(2)
    with col1:
        depth = st.slider("Crawl depth", 0, 2, 0)
    with col2:
        same_domain_only = st.toggle("Same-domain only", True)
    render_js = st.toggle("Render JavaScript (Playwright)", False)
    respect_robots = st.toggle("Respect robots.txt", True)
    timeout = st.number_input("Timeout (sec)", 5, 120, 25)
    user_agent = st.text_input("User-Agent", value=DEFAULT_UA)
    st.divider()
    st.subheader("LLM (Optional)")
    st.caption("Set OPENAI_API_KEY to enable summarization.")
    llm_model = st.text_input("OpenAI model", value="gpt-4o-mini")
    run = st.button("🚀 Start Crawl", type="primary", use_container_width=True)

if run and not url:
    st.error("Please enter a URL.")

if run and url:
    with st.spinner("Crawling..."):
        started = time.time()
        result = crawl_web(
            url=url,
            render_js=render_js,
            respect_robots=respect_robots,
            timeout=int(timeout),
            user_agent=user_agent,
            depth=int(depth),
            same_domain_only=bool(same_domain_only),
            llm_model=llm_model,
        )
        elapsed = time.time() - started

    st.success(f"Done in {elapsed:.2f}s")

    tabs = st.tabs(["Overview", "Pages", "Links", "Metadata", "Download"])

    with tabs[0]:
        st.subheader("Overview")
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Pages crawled", len(result.get('pages', [])))
        colB.metric("Depth", result.get('depth', 0))
        colC.metric("JS rendering", "On" if result.get('render_js') else "Off")
        colD.metric("Robots respected", "Yes" if result.get('respect_robots') else "No")

        if result.get('summary'):
            st.subheader("LLM Summary")
            st.json(result['summary'], expanded=False)

        if result.get('errors'):
            with st.expander("Warnings / Errors"):
                for e in result['errors']:
                    st.error(e)

    with tabs[1]:
        st.subheader("Pages")
        for i, page in enumerate(result.get('pages', []), start=1):
            with st.expander(f"{i}. {page.get('final_url')}  —  {page.get('title') or ''}"):
                cols = st.columns(4)
                cols[0].write(f"**Status**: {page.get('status_code')}")
                cols[1].write(f"**Content-Type**: {page.get('content_type')}")
                cols[2].write(f"**Text length**: {len(page.get('text') or ''):,}")
                cols[3].write(f"**Links**: {len(page.get('links') or [])}")

                sub_tabs = st.tabs(["Text", "Metadata", "Links", "Images"]) 
                with sub_tabs[0]:
                    st.text_area("Extracted Text", page.get('text') or '', height=300)
                with sub_tabs[1]:
                    st.json(page.get('metadata') or {}, expanded=False)
                with sub_tabs[2]:
                    links = page.get('links') or []
                    if links:
                        st.dataframe(links, use_container_width=True)
                    else:
                        st.info("No links found.")
                with sub_tabs[3]:
                    images = page.get('images') or []
                    if images:
                        st.dataframe(images, use_container_width=True)
                    else:
                        st.info("No images found.")

    with tabs[2]:
        st.subheader("All Links (First Page)")
        if result.get('pages'):
            links = result['pages'][0].get('links') or []
            st.dataframe(links, use_container_width=True)
        else:
            st.info("No page data available.")

    with tabs[3]:
        st.subheader("Metadata (First Page)")
        if result.get('pages'):
            st.json(result['pages'][0].get('metadata') or {}, expanded=False)
        else:
            st.info("No page data available.")

    with tabs[4]:
        st.subheader("Download Results")
        json_bytes = json.dumps(result, ensure_ascii=False, indent=2).encode('utf-8')
        st.download_button("⬇️ Download JSON", data=json_bytes, file_name="crawl_result.json", mime="application/json")

        import csv
        from io import StringIO
        csv_buf = StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=["href", "text", "internal"]) 
        writer.writeheader()
        if result.get('pages'):
            for lk in (result['pages'][0].get('links') or []):
                writer.writerow({k: lk.get(k, '') for k in ["href", "text", "internal"]})
        st.download_button("⬇️ Download Links CSV", data=csv_buf.getvalue().encode('utf-8'), file_name="links.csv", mime="text/csv")

else:
    st.info("Enter a URL and click **Start Crawl** from the sidebar.")
