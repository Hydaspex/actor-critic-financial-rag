from acfr.rag.controller import RagController
from acfr.sources.loader import load_sample_sources

c = RagController()
c.ingest(load_sample_sources())
print(c.query("What changed in revenue and liquidity?", top_k=3))
