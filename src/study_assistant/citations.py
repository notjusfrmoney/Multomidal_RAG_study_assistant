from .store import SearchResult


def format_citation(result: SearchResult) -> str:
    return f"{result.source_file}, page {result.page_start}"


def build_context(results: list[SearchResult]) -> str:
    return "\n\n".join(f"[{format_citation(result)}]\n{result.text}" for result in results)
