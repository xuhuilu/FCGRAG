# Subject Systems: Version Anchors

The three subject systems are third-party MIT-licensed projects and are not
redistributed in this repository. The benchmark and call graphs were built
from local source mirrors; because the mirrors carry no git metadata, each
tree is anchored by a content hash (SHA-256 of the sorted per-file SHA-256
manifest, computed over the mirror used for benchmark construction):

| System | Mirror dir | Upstream | License | Tree hash (SHA-256) |
|---|---|---|---|---|
| LangChain (Python) | `langchain-master` | https://github.com/langchain-ai/langchain | MIT | `8130b4d50b5d79c1c6377da5b66d71feaf8bc07cc59d02fd028155f356591716` |
| Ollama (C/C++/Go) | `ollama-main` | https://github.com/ollama/ollama | MIT | `9470dcd106b249f512591fe7e5857f1f8e821695748946d93ab4d8a10911e65c` |
| OpenClaw (TS/JS) | `openclaw-main` | https://github.com/openclaw/openclaw | MIT | `2ee9761f0eb228d6304d5b456b0af5f932f79885a0c0f76dcee5ab5218fc4e39` |

To reproduce from source, clone the upstream repositories and check out the
`master`/`main` snapshot matching these trees as closely as possible; exact
bit-equality can be verified by recomputing the tree hash with:

```bash
cd <source-tree> && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum
```

Exact snapshot archives of the three mirrors are also provided in the Zenodo
data record (https://doi.org/10.5281/zenodo.22128120) for bit-identical
reproduction.

Known version drift: five global-benchmark items reference documentation that
changed after the snapshot date; they are disclosed in the paper's
provenance/limitations sections. The OpenClaw benchmark was built from a
source zip snapshot (no public git history at construction time).
