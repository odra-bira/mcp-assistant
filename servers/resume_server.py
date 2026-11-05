"""
Resume MCP server.

Exposes the user's resume as a read-only tool and resource. Deliberately scoped to a
single document rather than general filesystem access (least privilege).

Lookup order in data/: resume.pdf, resume.txt, resume.md, then sample_resume.md.
Personal resume files are git-ignored; only the sample is committed.
"""

from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

server = MCPServer("resume")
DATA = Path(__file__).resolve().parent.parent / "data"


def load_resume() -> str:
    pdf = DATA / "resume.pdf"
    if pdf.exists():
        from pypdf import PdfReader

        text = "\n".join(page.extract_text() or "" for page in PdfReader(pdf).pages).strip()
        if text:
            return text
    for name in ("resume.txt", "resume.md", "sample_resume.md"):
        path = DATA / name
        if path.exists():
            return path.read_text(encoding="utf-8")
    return "No resume found. Add resume.pdf, resume.txt or resume.md to the data/ folder."


@server.tool(annotations=ToolAnnotations(read_only_hint=True))
def read_resume() -> str:
    """Return the full text of the user's resume."""
    return load_resume()


@server.resource("resume://current", mime_type="text/plain")
def resume_resource() -> str:
    """The user's current resume as plain text."""
    return load_resume()


if __name__ == "__main__":
    server.run()
