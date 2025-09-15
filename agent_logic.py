
"""
Agentic Web Crawler logic with defensive imports so the module always imports cleanly.
- Robust fetching via httpx + retries
- Optional Playwright rendering
- robots.txt (toggle)
- Content-type aware parsing: HTML, PDF, DOCX, TXT
- LangChain tools imported lazily with graceful fallbacks
"""

import re
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
import tldextract
from charset_normalizer import from_bytes
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Optional heavy deps (handled defensively)
try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None
try:
    import docx
except Exception:  # pragma: no cover
    docx = None
try:
    from playwright.sync_api import sync_playwright
except Exception:  # pragma: no cover
    sync_playwright = None

# LangChain defensive import
try:
    from langchain.tools import tool
except Exception:  # pragma: no cover
    # Fallback no-op decorator so code still runs
    def tool(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatPromptTemplate = None
    ChatOpenAI = None

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

@dataclass
class PageResult:
    url: str
    final_url: str
    status_code: int
    content_type: str
    title: Optional[str]
    text: str
    metadata: Dict[str, str]
    links: List[Dict[str, str]]
    images: List[Dict[str, str]]
    errors: List[str]


def _same_domain(u1: str, u2: str) -> bool:
    e1, e2 = tldextract.extract(u1), tldextract.extract(u2)
    return (e1.domain, e1.suffix) == (e2.domain, e2.suffix)


def _normalize_text_bytes(content: bytes, fallback_encoding: str = 'utf-8') -> str:
    try:
        best = from_bytes(content).best()
        if best is None:
            return content.decode(fallback_encoding, errors='replace')
        return str(best)
    except Exception:
        return content.decode(fallback_encoding, errors='replace')


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8),
       retry=retry_if_exception_type(httpx.HTTPError))
def fetch_url(url: str, timeout: int = 25, user_agent: str = DEFAULT_UA,
              allow_redirects: bool = True) -> Tuple[str, int, Dict[str, str], bytes]:
    headers = {
        'User-Agent': user_agent,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    with httpx.Client(timeout=timeout, follow_redirects=allow_redirects, http2=True, verify=True) as client:
        r = client.get(url, headers=headers)
        r.raise_for_status()
        return str(r.url), r.status_code, dict(r.headers), r.content


def render_with_playwright(url: str, timeout: int = 30, user_agent: str = DEFAULT_UA) -> Tuple[str, int, Dict[str, str], bytes]:
    if sync_playwright is None:
        raise RuntimeError("Playwright not installed. Run: playwright install")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=user_agent)
        page = context.new_page()
        page.set_default_timeout(timeout * 1000)
        page.goto(url)
        page.wait_for_load_state('networkidle')
        html = page.content().encode('utf-8', errors='ignore')
        final = page.url
        headers = {'content-type': 'text/html; charset=utf-8'}
        browser.close()
        return final, 200, headers, html


def parse_html(base_url: str, html: bytes):
    text = ''
    title = None
    metadata: Dict[str, str] = {}
    links: List[Dict[str, str]] = []
    images: List[Dict[str, str]] = []

    try:
        try:
            import trafilatura  # lazy
            text = trafilatura.extract(html, url=base_url) or ''
        except Exception:
            text = ''
        soup = BeautifulSoup(html, 'lxml')
        if not text:
            text = soup.get_text("\n", strip=True)
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        for meta in soup.find_all('meta'):
            k = meta.get('name') or meta.get('property')
            v = meta.get('content')
            if k and v:
                metadata[k.strip()] = v.strip()
        for a in soup.find_all('a', href=True):
            href = urljoin(base_url, a['href'])
            txt = (a.get_text(strip=True) or '')[:200]
            links.append({'href': href, 'text': txt, 'internal': str(_same_domain(base_url, href)).lower()})
        for img in soup.find_all('img'):
            src = img.get('src')
            if not src:
                continue
            src_abs = urljoin(base_url, src)
            images.append({'src': src_abs, 'alt': img.get('alt', '')})
    except Exception as e:
        metadata.setdefault('parse_error', str(e))

    return title or '', metadata, links, images


def parse_pdf(content: bytes) -> str:
    if not fitz:
        return ''
    try:
        with fitz.open(stream=content, filetype='pdf') as doc:
            return '
'.join(page.get_text() for page in doc)
    except Exception:
        return ''


def parse_docx(content: bytes) -> str:
    if not docx:
        return ''
    try:
        from io import BytesIO
        d = docx.Document(BytesIO(content))
        return '
'.join(p.text for p in d.paragraphs)
    except Exception:
        return ''


def is_allowed_by_robots(url: str, user_agent: str = DEFAULT_UA) -> bool:
    try:
        from urllib import robotparser
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True


@tool("fetch_page", return_direct=False)
def tool_fetch_page(url: str, render_js: bool = False, timeout: int = 25, user_agent: str = DEFAULT_UA) -> dict:
    """Fetch a URL (static or JS-rendered). Returns content_text + content_bytes_b64."""
    import base64
    try:
        if render_js:
            final, status, headers, content = render_with_playwright(url, timeout=timeout, user_agent=user_agent)
        else:
            final, status, headers, content = fetch_url(url, timeout=timeout, user_agent=user_agent)
        ctype = headers.get('content-type', '').lower()
        text = _normalize_text_bytes(content)
        b64 = base64.b64encode(content).decode('ascii')
        return {
            'final_url': final,
            'status_code': status,
            'headers': headers,
            'content_text': text,
            'content_bytes_b64': b64,
            'content_type': ctype,
        }
    except Exception as e:
        return {'error': str(e)}


@tool("extract_from_html", return_direct=False)
def tool_extract_from_html(base_url: str, html_text: str) -> dict:
    title, metadata, links, images = parse_html(base_url, html_text.encode('utf-8', errors='ignore'))
    try:
        import trafilatura
        main_text = trafilatura.extract(html_text, url=base_url) or ''
    except Exception:
        main_text = ''
    if not main_text:
        main_text = BeautifulSoup(html_text, 'lxml').get_text("
", strip=True)
    return {'title': title, 'metadata': metadata, 'links': links, 'images': images, 'text': main_text}


@tool("sniff_and_parse_bytes", return_direct=False)
def tool_sniff_and_parse_bytes(content_type: str, content_bytes_b64: str) -> dict:
    import base64
    raw = base64.b64decode(content_bytes_b64.encode('ascii')) if content_bytes_b64 else b''
    ctype = (content_type or '').split(';')[0].strip().lower()
    text = ''
    if 'pdf' in ctype:
        text = parse_pdf(raw)
    elif 'word' in ctype or ctype in {'application/vnd.openxmlformats-officedocument.wordprocessingml.document','application/msword'}:
        text = parse_docx(raw)
    elif ctype.startswith('text/') or ctype in {'application/json', 'application/xml'}:
        text = _normalize_text_bytes(raw)
    else:
        text = _normalize_text_bytes(raw)
    return {'text': text}


if ChatPromptTemplate is not None and ChatOpenAI is not None:
    SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "You are an expert analyst. Given raw extracted web page content, produce a concise JSON with keys: 'title', 'key_points' (3-7 bullets), 'entities', 'language', 'tags'. Keep it short and objective."),
        ("human", "URL: {url}
TITLE: {title}
TEXT (truncated):
{snippet}")
    ])
else:
    SUMMARY_PROMPT = None


def summarize_with_llm(url: str, title: str, text: str, model: str = 'gpt-4o-mini') -> Optional[dict]:
    if SUMMARY_PROMPT is None or ChatOpenAI is None or not text:
        return None
    llm = ChatOpenAI(model=model, temperature=0.2)
    snippet = text[:6000]
    chain = SUMMARY_PROMPT | llm
    try:
        resp = chain.invoke({"url": url, "title": title, "snippet": snippet})
        content = getattr(resp, 'content', None)
        if isinstance(content, str):
            # Try to find a JSON block at the end
            m = re.search(r"\{[\s\S]*\}$", content.strip())
            if m:
                import json
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
            return {"summary": content.strip()}
        return None
    except Exception:
        return None


def crawl_web(
    url: str,
    render_js: bool = False,
    respect_robots: bool = True,
    timeout: int = 25,
    user_agent: str = DEFAULT_UA,
    depth: int = 0,
    same_domain_only: bool = True,
    llm_model: str = 'gpt-4o-mini'
) -> Dict:
    visited = set()
    queue: List[Tuple[str, int]] = [(url, 0)]
    pages: List[PageResult] = []
    errors: List[str] = []

    while queue:
        current, d = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        if respect_robots and not is_allowed_by_robots(current, user_agent=user_agent):
            errors.append(f"Blocked by robots.txt: {current}")
            continue

        try:
            fetched = tool_fetch_page.invoke({'url': current, 'render_js': render_js, 'timeout': timeout, 'user_agent': user_agent})
        except Exception as e:
            errors.append(f"Fetch failed for {current}: {e}")
            continue

        if 'error' in fetched:
            errors.append(f"Fetch failed for {current}: {fetched['error']}")
            continue

        final = fetched['final_url']
        status = fetched['status_code']
        headers = fetched.get('headers', {})
        ctype = headers.get('content-type', fetched.get('content_type', '')).lower()
        content_text = fetched.get('content_text', '')
        content_b64 = fetched.get('content_bytes_b64', '')

        title = ''
        text = ''
        metadata: Dict[str, str] = {}
        links: List[Dict[str, str]] = []
        images: List[Dict[str, str]] = []
        page_errors: List[str] = []

        try:
            if 'html' in ctype:
                extracted = tool_extract_from_html.invoke({'base_url': final, 'html_text': content_text})
                title = extracted.get('title') or ''
                metadata = extracted.get('metadata') or {}
                links = extracted.get('links') or []
                images = extracted.get('images') or []
                text = extracted.get('text') or ''
            else:
                parsed = tool_sniff_and_parse_bytes.invoke({'content_type': ctype, 'content_bytes_b64': content_b64})
                text = parsed.get('text') or ''
        except Exception as e:
            page_errors.append(str(e))

        pages.append(PageResult(
            url=current,
            final_url=final,
            status_code=status,
            content_type=ctype or 'unknown',
            title=title or None,
            text=text or '',
            metadata=metadata,
            links=links,
            images=images,
            errors=page_errors,
        ))

        if d < depth and links:
            for lk in links:
                href = lk.get('href')
                if not href:
                    continue
                if same_domain_only and not _same_domain(url, href):
                    continue
                if href not in visited:
                    queue.append((href, d + 1))

    summary = None
    if pages:
        try:
            summary = summarize_with_llm(pages[0].final_url, pages[0].title or '', pages[0].text, model=llm_model)
        except Exception:
            summary = None

    return {
        'input_url': url,
        'depth': depth,
        'render_js': render_js,
        'respect_robots': respect_robots,
        'same_domain_only': same_domain_only,
        'pages': [asdict(p) for p in pages],
        'summary': summary,
        'errors': errors,
        'timestamp': time.time(),
    }

