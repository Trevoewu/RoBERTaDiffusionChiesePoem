import argparse
import json
import logging
import re
import time
from collections import OrderedDict
from html import unescape
from html.parser import HTMLParser
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin

import requests


BASE_URL = "https://www.shigeku.com/xlib/xd/zgsg/"
INDEX_URL = urljoin(BASE_URL, "index.htm")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


class IndexLinkParser(HTMLParser):
    """Extract author page links and names from the index page."""

    def __init__(self) -> None:
        super().__init__()
        self._current_href: Optional[str] = None
        self._capture_text: bool = False
        self.links: List[Tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str]]) -> None:
        if tag.lower() != "a":
            return

        attr_map: Dict[str, str] = dict(attrs)
        href = attr_map.get("href", "").strip()
        if not href or href.lower().startswith("http"):
            return

        if not href.lower().endswith((".htm", ".html")):
            return

        if href.lower() == "index.htm":
            return

        self._current_href = href
        self._capture_text = True

    def handle_data(self, data: str) -> None:
        if not self._capture_text or self._current_href is None:
            return

        text = data.strip()
        if text:
            self.links.append((self._current_href, text))

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a":
            self._current_href = None
            self._capture_text = False


class PoemPageParser(HTMLParser):
    """Parse poem titles and content from an author page."""

    def __init__(self) -> None:
        super().__init__()
        self.poems: List[Dict[str, str]] = []
        self._current_poem: Optional[Dict[str, List[str]]] = None
        self._current_align: Optional[str] = None
        self._capture_title = False
        self._capture_content = False
        self._current_paragraph_has_text = False

    # --- Helpers -----------------------------------------------------------------

    def _append_text(self, text: str) -> None:
        if not self._current_poem:
            return

        if self._capture_title:
            self._current_poem.setdefault("title_parts", []).append(text)
        elif self._capture_content:
            self._current_poem.setdefault("content_parts", []).append(text)
            if text.strip():
                self._current_paragraph_has_text = True

    def _ensure_poem(self) -> None:
        if self._current_poem is None:
            self._current_poem = {"title_parts": [], "content_parts": []}

    def _finalize_current_poem(self) -> None:
        if not self._current_poem:
            return

        title_raw = "".join(self._current_poem.get("title_parts", []))
        content_raw = "".join(self._current_poem.get("content_parts", []))

        title = unescape(title_raw).replace("\xa0", " ").strip()
        content = unescape(content_raw).replace("\xa0", " ")
        content = content.replace("\r", "")

        normalized_lines: List[str] = []
        for raw_line in content.splitlines():
            line = raw_line.rstrip(" \t")
            line = line.lstrip(" \t")
            normalized_lines.append(line)

        content = "\n".join(normalized_lines)
        content = re.sub(r"\n{3,}", "\n\n", content).strip("\n")

        if title and content:
            self.poems.append({"title": title, "content": content})

        self._current_poem = None
        self._capture_title = False
        self._capture_content = False
        self._current_paragraph_has_text = False

    # --- HTMLParser overrides -----------------------------------------------------

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str]]) -> None:
        tag_lower = tag.lower()

        if tag_lower == "p":
            attr_map = {key.lower(): value for key, value in attrs}
            self._current_align = attr_map.get("align", "").lower()

            if self._current_poem and self._current_align != "center":
                self._capture_content = True
                if self._current_poem.setdefault("content_parts", []):
                    self._current_poem["content_parts"].append("\n")
                self._current_paragraph_has_text = False
            else:
                self._capture_content = False

        elif tag_lower == "a":
            attr_map = {key.lower(): value for key, value in attrs}
            if attr_map.get("name") and self._current_align == "center":
                self._finalize_current_poem()
                self._ensure_poem()
                self._capture_title = True

        elif tag_lower == "br":
            if self._capture_content and self._current_poem:
                self._current_poem.setdefault("content_parts", []).append("\n")

        elif tag_lower == "hr":
            self._finalize_current_poem()

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()

        if tag_lower == "p":
            if self._capture_title:
                self._capture_title = False
            elif self._capture_content and self._current_poem:
                if self._current_paragraph_has_text:
                    self._current_poem.setdefault("content_parts", []).append("\n")
                self._capture_content = False
                self._current_paragraph_has_text = False

        elif tag_lower == "body":
            self._finalize_current_poem()

    def handle_data(self, data: str) -> None:
        self._append_text(data)

    def handle_entityref(self, name: str) -> None:
        self._append_text(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self._append_text(f"&#{name};")

    def close(self) -> None:  # type: ignore[override]
        super().close()
        self._finalize_current_poem()


def fetch_text(session: requests.Session, url: str) -> str:
    response = session.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()
    return response.content.decode("gb18030", errors="ignore")


def parse_index(index_html: str) -> OrderedDict:
    parser = IndexLinkParser()
    parser.feed(index_html)
    parser.close()

    ordered: OrderedDict[str, str] = OrderedDict()
    for href, name in parser.links:
        if href not in ordered:
            ordered[href] = name
    return ordered


def parse_poems(poet_html: str) -> List[Dict[str, str]]:
    parser = PoemPageParser()
    parser.feed(poet_html)
    parser.close()
    return parser.poems


def iter_poems(session: requests.Session,
               author_map: OrderedDict,
               delay: float) -> Iterable[Tuple[str, Dict[str, str]]]:
    for href, author in author_map.items():
        author_url = urljoin(BASE_URL, href)
        try:
            html_text = fetch_text(session, author_url)
        except requests.RequestException as exc:
            logging.warning("Failed to fetch %s: %s", author_url, exc)
            continue

        poems = parse_poems(html_text)
        if not poems:
            logging.debug("No poems parsed for %s", author_url)
            continue

        for poem in poems:
            yield author, poem

        if delay > 0:
            time.sleep(delay)


def run(output_path: str,
        max_authors: Optional[int],
        max_poems: Optional[int],
        delay: float) -> None:
    session = requests.Session()

    try:
        index_html = fetch_text(session, INDEX_URL)
    except requests.RequestException as exc:
        raise SystemExit(f"Failed to fetch index page: {exc}") from exc

    author_map = parse_index(index_html)
    if max_authors is not None:
        items = list(author_map.items())[:max_authors]
        author_map = OrderedDict(items)

    total = 0
    with open(output_path, "w", encoding="utf-8") as outfile:
        for author, poem in iter_poems(session, author_map, delay):
            record = {
                "title": poem["title"],
                "author": author,
                "content": poem["content"],
            }
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1

            if max_poems is not None and total >= max_poems:
                logging.info("Reached user-defined maximum of %s poems", max_poems)
                break

    logging.info("Saved %s poems to %s", total, output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crawl modern Chinese poems from shigeku.com and export JSONL"
    )
    parser.add_argument(
        "--output",
        default="modern_poems.jsonl",
        help="Output JSONL file path (default: modern_poems.jsonl)",
    )
    parser.add_argument(
        "--max-authors",
        type=int,
        default=None,
        help="Optionally limit the number of author pages to crawl",
    )
    parser.add_argument(
        "--max-poems",
        type=int,
        default=None,
        help="Optionally limit the total number of poems to export",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between author page requests (default: 0.5)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity (default: INFO)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s:%(message)s",
    )

    run(
        output_path=args.output,
        max_authors=args.max_authors,
        max_poems=args.max_poems,
        delay=max(args.delay, 0.0),
    )


if __name__ == "__main__":
    main()
