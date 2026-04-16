"""
Search anime clips using text queries — Qwen3-VL-Embedding-2B

Usage:
    python search.py "man holding apple"
    python search.py "rainy city at night" --top 10
    python search.py "explosion" --namespace anime-clips

The same model that embedded the video frames also embeds your text query,
so text and visual vectors live in the same space — cross-modal search.
"""

import argparse
import sys
import os
import time

import torch

# ── Config ────────────────────────────────────────────────────
TURBOPUFFER_API_KEY = os.environ.get("TURBOPUFFER_API_KEY", "")
MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"

# Global model — loaded once, reused for interactive mode
_embedder = None


def load_model():
    global _embedder
    if _embedder is not None:
        return _embedder

    print(f"Loading {MODEL_NAME}...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    from qwen3_vl_embedding import Qwen3VLEmbedder

    _embedder = Qwen3VLEmbedder(
        model_name_or_path=MODEL_NAME,
        torch_dtype=torch.float16,
    )
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB\n")
    return _embedder


def embed_query(text: str) -> list[float]:
    embedder = load_model()
    inputs = [{"text": text, "instruction": "Find a video clip that matches this description."}]
    embeddings = embedder.process(inputs)
    return embeddings[0].cpu().numpy().tolist()


def search(query: str, namespace: str = "anime-clips", top_k: int = 5):
    import turbopuffer as tpuf

    # Embed the query
    t0 = time.time()
    query_vector = embed_query(query)
    embed_time = time.time() - t0

    # Search turbopuffer
    client = tpuf.Turbopuffer(api_key=TURBOPUFFER_API_KEY, region="gcp-us-central1")
    ns = client.namespace(namespace)

    t1 = time.time()
    response = ns.query(
        rank_by=["vector", "ANN", query_vector],
        top_k=top_k,
        distance_metric="cosine_distance",
        include_attributes=["source", "category", "video_id", "path", "timestamp"],
    )
    search_time = time.time() - t1

    return response.rows, embed_time, search_time


def main():
    parser = argparse.ArgumentParser(description="Search anime clips with text")
    parser.add_argument("query", nargs="?", help="Search query text")
    parser.add_argument("--top", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--namespace", default="anime-clips", help="Turbopuffer namespace")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive search mode")
    args = parser.parse_args()

    if args.interactive or not args.query:
        # Interactive mode — keep model loaded
        load_model()
        print("Interactive search mode. Type a query and press Enter. Ctrl+C to exit.\n")
        while True:
            try:
                query = input("Search> ").strip()
                if not query:
                    continue
                results, embed_time, search_time = search(query, args.namespace, args.top)
                print(f"\n  Query embedded in {embed_time*1000:.0f}ms, search in {search_time*1000:.0f}ms")
                print(f"  Top {len(results)} results for: \"{query}\"\n")
                for i, row in enumerate(results):
                    d = row.to_dict()
                    dist = d.get('$dist', 1.0)
                    print(f"  {i+1}. [{1-dist:.3f}] {d.get('category', '?')} / {d.get('video_id', '?')} @ {d.get('timestamp', '?')}s")
                    print(f"     {d.get('path', 'no path')}")
                print()
            except KeyboardInterrupt:
                print("\nBye.")
                break
    else:
        results, embed_time, search_time = search(args.query, args.namespace, args.top)
        print(f"\nQuery embedded in {embed_time*1000:.0f}ms, search in {search_time*1000:.0f}ms")
        print(f"Top {len(results)} results for: \"{args.query}\"\n")
        for i, row in enumerate(results):
            d = row.to_dict()
            dist = d.get('$dist', 1.0)
            print(f"  {i+1}. [similarity={1-dist:.3f}] {d.get('category', '?')} / {d.get('video_id', '?')} @ {d.get('timestamp', '?')}s")
            print(f"     {d.get('path', 'no path')}")


if __name__ == "__main__":
    main()
