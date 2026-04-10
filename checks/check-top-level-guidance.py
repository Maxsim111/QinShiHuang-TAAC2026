from __future__ import annotations

import json
import re
import sys
from pathlib import Path


PACK_FILES = ["AGENTS.md", "README.md"]
REQUIRED_PATHS = [
    "README.md",
    "docs/project.md",
    "docs/log.md",
]
CANONICAL_AUTHORITY = {
    "project context": "docs/project.md",
    "work log": "docs/log.md",
}


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text.strip())) if text.strip() else 0


def main() -> int:
    repo_root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    line_records = []
    total_words = 0

    for relative_path in PACK_FILES:
        lines = (repo_root / relative_path).read_text(encoding="utf-8").splitlines()
        total_words += word_count("\n".join(lines))
        for index, line in enumerate(lines, start=1):
            line_records.append({"file": relative_path, "line_number": index, "text": line})

    path_mentions = []
    for record in line_records:
        for match in re.finditer(r"`([^`]+\.md)`", record["text"]):
            relative_path = match.group(1)
            path_mentions.append(
                {
                    "path": relative_path,
                    "exists": (repo_root / relative_path).exists(),
                    "file": record["file"],
                    "line_number": record["line_number"],
                }
            )

    stale_references = sorted({mention["path"] for mention in path_mentions if not mention["exists"]})

    required_path_coverage = []
    for required_path in REQUIRED_PATHS:
        mentions = [mention for mention in path_mentions if mention["path"] == required_path]
        required_path_coverage.append(
            {
                "path": required_path,
                "mentioned_in_pack": bool(mentions),
                "mention_count": len(mentions),
            }
        )

    authority_claims = []
    for record in line_records:
        match = re.search(r"Source of truth for (?P<topic>[^:]+): `(?P<path>[^`]+\.md)`", record["text"], re.IGNORECASE)
        if not match:
            continue
        topic = match.group("topic").strip().lower()
        path = match.group("path")
        authority_claims.append(
            {
                "topic": topic,
                "path": path,
                "expected_path": CANONICAL_AUTHORITY.get(topic),
                "is_expected": CANONICAL_AUTHORITY.get(topic) == path,
            }
        )

    wrong_authority_topics = sorted(
        {
            claim["topic"]
            for claim in authority_claims
            if claim["expected_path"] is not None and not claim["is_expected"]
        }
    )

    conflict_map: dict[str, set[str]] = {}
    for claim in authority_claims:
        conflict_map.setdefault(claim["topic"], set()).add(claim["path"])

    authority_conflicts = [
        {"topic": topic, "paths": sorted(paths)}
        for topic, paths in conflict_map.items()
        if len(paths) > 1
    ]

    covered_count = sum(1 for item in required_path_coverage if item["mentioned_in_pack"])
    verdict = "pass"
    if stale_references or wrong_authority_topics or authority_conflicts or covered_count != len(REQUIRED_PATHS):
        verdict = "fail"

    result = {
        "verdict": verdict,
        "summary": {
            "top_level_word_count": total_words,
            "required_paths_covered": covered_count,
            "required_path_count": len(REQUIRED_PATHS),
            "stale_reference_count": len(stale_references),
            "wrong_authority_count": len(wrong_authority_topics),
            "authority_conflict_count": len(authority_conflicts),
        },
        "stale_references": stale_references,
        "wrong_authority_topics": wrong_authority_topics,
        "authority_conflicts": authority_conflicts,
        "required_path_coverage": required_path_coverage,
    }
    print(json.dumps(result, indent=2))
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
