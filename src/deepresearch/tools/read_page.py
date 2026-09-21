"""Fetch a public web page and extract readable text for the agent."""

from __future__ import annotations

import ipaddress
import json
import re
import socket
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx
from langchain_core.tools import tool
from lxml import etree, html


DEFAULT_MAX_CHARS = 12_000
MAX_CHARS_LIMIT = 50_000
MAX_RESPONSE_BYTES = 2_000_000
MAX_REDIRECTS = 5
ALLOWED_CONTENT_TYPES = {"text/html", "text/plain", "application/xhtml+xml"}
USER_AGENT = "deepresearch-demo/0.1 (+local research agent)"
PROXY_FAKE_IP_NETWORK = ipaddress.ip_network("198.18.0.0/15")


def _error_payload(url: str, message: str) -> dict[str, Any]:
    return {"ok": False, "requested_url": url, "error": message}


def _resolved_addresses(hostname: str, port: int) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    return list({ipaddress.ip_address(info[4][0]) for info in infos})


def _is_public_address(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
    *,
    allow_proxy_fake_ip: bool = False,
) -> bool:
    if allow_proxy_fake_ip and address in PROXY_FAKE_IP_NETWORK:
        return True
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def validate_public_url(url: str) -> str | None:
    """Return an error message when a URL is unsafe or cannot be resolved."""

    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        return f"Invalid URL: {exc}"
    if parsed.scheme not in {"http", "https"}:
        return "Only http:// and https:// URLs are allowed"
    if not parsed.hostname:
        return "URL must include a hostname"
    if parsed.username or parsed.password:
        return "URLs containing credentials are not allowed"

    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return "Localhost URLs are not allowed"

    try:
        try:
            ipaddress.ip_address(hostname)
            hostname_is_ip_literal = True
        except ValueError:
            hostname_is_ip_literal = False
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = _resolved_addresses(hostname, port)
    except (OSError, ValueError) as exc:
        return f"Could not resolve URL hostname: {exc}"
    if not addresses:
        return "URL hostname did not resolve to an address"
    if any(
        not _is_public_address(
            address,
            allow_proxy_fake_ip=not hostname_is_ip_literal,
        )
        for address in addresses
    ):
        return "Private, local, reserved, and link-local addresses are not allowed"
    return None


def _extract_html_text(raw_html: str) -> tuple[str, str]:
    if not raw_html.strip():
        return "", ""
    try:
        document = html.fromstring(raw_html)
    except (etree.ParserError, ValueError):
        return "", re.sub(r"\s+", " ", raw_html).strip()
    for element in document.xpath(
        "//script|//style|//noscript|//svg|//nav|//footer|//header|//form"
    ):
        element.drop_tree()

    title_nodes = document.xpath("//title/text()")
    title = " ".join(str(part).strip() for part in title_nodes if str(part).strip())
    lines = []
    for line in document.text_content().splitlines():
        normalized = re.sub(r"\s+", " ", line).strip()
        if normalized:
            lines.append(normalized)
    return title, "\n".join(lines)


def _decode_body(response: httpx.Response, body: bytes) -> str:
    encoding = response.encoding or "utf-8"
    try:
        return body.decode(encoding, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def read_page_content(
    url: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    *,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Fetch and normalize one page; ``transport`` exists for deterministic tests."""

    requested_url = url.strip()
    if not requested_url:
        return _error_payload(url, "URL cannot be empty")
    if max_chars < 1:
        return _error_payload(requested_url, "max_chars must be at least 1")
    character_limit = min(max_chars, MAX_CHARS_LIMIT)

    current_url = requested_url
    try:
        with httpx.Client(
            headers={"User-Agent": USER_AGENT},
            follow_redirects=False,
            timeout=15,
            transport=transport,
        ) as client:
            for redirect_count in range(MAX_REDIRECTS + 1):
                validation_error = validate_public_url(current_url)
                if validation_error:
                    return _error_payload(requested_url, validation_error)

                with client.stream("GET", current_url) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            return _error_payload(
                                requested_url,
                                "Redirect response did not include a Location header",
                            )
                        if redirect_count == MAX_REDIRECTS:
                            return _error_payload(requested_url, "Too many redirects")
                        current_url = urljoin(str(response.url), location)
                        continue

                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "")
                    media_type = content_type.split(";", 1)[0].strip().lower()
                    if media_type and media_type not in ALLOWED_CONTENT_TYPES:
                        return _error_payload(
                            requested_url,
                            f"Unsupported content type: {media_type}",
                        )

                    chunks: list[bytes] = []
                    byte_count = 0
                    response_truncated = False
                    for chunk in response.iter_bytes():
                        remaining = MAX_RESPONSE_BYTES - byte_count
                        if remaining <= 0:
                            response_truncated = True
                            break
                        chunks.append(chunk[:remaining])
                        byte_count += min(len(chunk), remaining)
                        if len(chunk) > remaining:
                            response_truncated = True
                            break
                    body = b"".join(chunks)
                    raw_text = _decode_body(response, body)
                    if media_type in {"text/html", "application/xhtml+xml", ""}:
                        title, content = _extract_html_text(raw_text)
                    else:
                        title, content = "", raw_text.strip()

                    content_truncated = len(content) > character_limit
                    return {
                        "ok": True,
                        "requested_url": requested_url,
                        "final_url": str(response.url),
                        "title": title,
                        "content": content[:character_limit],
                        "truncated": response_truncated or content_truncated,
                    }
    except (httpx.HTTPError, OSError, ValueError) as exc:
        return _error_payload(requested_url, f"Page read failed: {exc}")

    return _error_payload(requested_url, "Page read ended unexpectedly")


@tool("read_page", parse_docstring=True)
def read_page_tool(url: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Read the visible text of a public web page found during research.

    Args:
        url: Public http or https page URL to read.
        max_chars: Maximum number of extracted text characters to return.
    """

    return json.dumps(
        read_page_content(url, max_chars=max_chars),
        ensure_ascii=False,
    )
