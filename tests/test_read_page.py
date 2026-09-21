import importlib
import ipaddress

import httpx

from deepresearch.tools.read_page import read_page_content, validate_public_url


read_page_module = importlib.import_module("deepresearch.tools.read_page")


def allow_public_dns(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        read_page_module,
        "_resolved_addresses",
        lambda hostname, port: [ipaddress.ip_address("93.184.216.34")],
    )


def test_url_validation_rejects_non_http_and_private_addresses(monkeypatch) -> None:  # noqa: ANN001
    assert "Only http" in validate_public_url("file:///etc/passwd")
    monkeypatch.setattr(
        read_page_module,
        "_resolved_addresses",
        lambda hostname, port: [ipaddress.ip_address("127.0.0.1")],
    )

    assert "not allowed" in validate_public_url("http://127.0.0.1/private")


def test_proxy_fake_ip_is_allowed_only_for_domain_names(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        read_page_module,
        "_resolved_addresses",
        lambda hostname, port: [ipaddress.ip_address("198.18.0.49")],
    )

    assert validate_public_url("https://docs.example.com/page") is None
    assert "not allowed" in validate_public_url("http://198.18.0.49/private")


def test_read_page_extracts_visible_html_text(monkeypatch) -> None:  # noqa: ANN001
    allow_public_dns(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="""
                <html><head><title>Example Page</title>
                <style>.hidden { display: none; }</style></head>
                <body><nav>Navigation</nav><main>
                <h1>Research heading</h1><p>Useful evidence.</p>
                <script>alert('ignore me')</script></main></body></html>
            """,
        )

    result = read_page_content(
        "https://example.com/article",
        transport=httpx.MockTransport(handler),
    )

    assert result["ok"] is True
    assert result["title"] == "Example Page"
    assert "Research heading" in result["content"]
    assert "Useful evidence" in result["content"]
    assert "Navigation" not in result["content"]
    assert "alert" not in result["content"]
    assert ".hidden" not in result["content"]


def test_read_page_follows_checked_redirect_and_truncates(monkeypatch) -> None:  # noqa: ANN001
    allow_public_dns(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "example.com":
            return httpx.Response(302, headers={"location": "https://docs.example.org/page"})
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            text="abcdefghijklmnopqrstuvwxyz",
        )

    result = read_page_content(
        "https://example.com/start",
        max_chars=10,
        transport=httpx.MockTransport(handler),
    )

    assert result["ok"] is True
    assert result["final_url"] == "https://docs.example.org/page"
    assert result["content"] == "abcdefghij"
    assert result["truncated"] is True


def test_read_page_rejects_unsupported_content_type(monkeypatch) -> None:  # noqa: ANN001
    allow_public_dns(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/pdf"},
            content=b"%PDF",
        )

    result = read_page_content(
        "https://example.com/report.pdf",
        transport=httpx.MockTransport(handler),
    )

    assert result["ok"] is False
    assert result["error"] == "Unsupported content type: application/pdf"


def test_read_page_handles_empty_html(monkeypatch) -> None:  # noqa: ANN001
    allow_public_dns(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"",
        )

    result = read_page_content(
        "https://example.com/empty",
        transport=httpx.MockTransport(handler),
    )

    assert result["ok"] is True
    assert result["title"] == ""
    assert result["content"] == ""
