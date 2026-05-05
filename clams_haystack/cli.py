"""Command line interface for the Haystack video catalog agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import HaystackVideoAgent
from .repository import CatalogRepository


def main() -> None:
    parser = argparse.ArgumentParser(description="Haystack video catalog/QA agent")
    parser.add_argument("--catalog-root", default="data/haystack_catalog")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Register a raw video and optional index")
    ingest.add_argument("video_path")
    ingest.add_argument("--video-id")
    ingest.add_argument("--index-path")
    ingest.add_argument("--metadata-json", default="{}")

    overview = sub.add_parser("overview", help="Show catalog state for a video")
    overview.add_argument("video_id")

    search = sub.add_parser("search", help="Lexically search catalog/content documents")
    search.add_argument("query")
    search.add_argument("--video-id")
    search.add_argument("--layer")
    search.add_argument("--top-k", type=int, default=8)

    ask = sub.add_parser("ask", help="Ask with the Haystack RAG pipeline")
    ask.add_argument("question")
    ask.add_argument("--video-id")

    agent_cmd = sub.add_parser("agent", help="Run the Haystack tool agent")
    agent_cmd.add_argument("message")
    agent_cmd.add_argument("--no-searchable-tools", action="store_true")

    update = sub.add_parser("update-metadata", help="Store a catalog metadata value")
    update.add_argument("video_id")
    update.add_argument("field")
    update.add_argument("value")
    update.add_argument("--source", default="user")

    args = parser.parse_args()
    repo = CatalogRepository(args.catalog_root)
    agent = HaystackVideoAgent(repository=repo)

    if args.command == "ingest":
        record = agent.register_video(
            video_path=Path(args.video_path),
            supplied_metadata=json.loads(args.metadata_json),
            index_path=args.index_path,
            video_id=args.video_id,
        )
        print(json.dumps(record.to_dict(), indent=2))
        questions = agent.missing_catalog_questions(record.video_id)
        if questions:
            print("\nMissing catalog questions:")
            for q in questions:
                print(f"- {q}")
    elif args.command == "overview":
        print(json.dumps(agent.get_video_overview(args.video_id), indent=2))
    elif args.command == "search":
        print(agent.search_content(
            query=args.query,
            video_id=args.video_id,
            layer=args.layer,
            top_k=args.top_k,
        )["formatted"])
    elif args.command == "ask":
        print(agent.ask_with_rag(args.question, video_id=args.video_id))
    elif args.command == "agent":
        print(agent.run_tool_agent(
            args.message,
            searchable=not args.no_searchable_tools,
        ))
    elif args.command == "update-metadata":
        print(json.dumps(agent.update_catalog_metadata(
            args.video_id,
            args.field,
            args.value,
            source=args.source,
        ), indent=2))


if __name__ == "__main__":
    main()
