
# 🕷️ Agentic Web Crawler (LangChain + Streamlit)

This is a **defensive** build that avoids import-time crashes on Streamlit Cloud by guarding optional imports. If LangChain/OpenAI/Playwright aren't available, the app still runs (with reduced features).

See the sidebar for toggles. For JS-heavy sites, run `playwright install` and enable **Render JavaScript**.
