# search-colbert

Late-interaction **first-stage** retriever (ColBERT / MaxSim) for BrainAPI `POST /retrieve/search`. It keeps its own token-embedding index (plugin-local, in memory) and registers channel `plugin:colbert`. It does **not** run on `/retrieve/context`. Core hybrid works if this plugin is absent.

Unknown or missing `plugin:colbert` is **400**, never treated as a ranking miss.

| | |
|---|---|
| Registry name | `search-colbert` |
| Version | `0.1.0` |
| BrainAPI | `>=2.17.0` |
| Channel | `plugin:colbert` |
| Default model | `colbert-ir/colbertv2.0` |
| Index | `POST /search-colbert/index` |
| Health | `GET /search-colbert/health` |

## Install

```bash
git clone https://github.com/Lumen-Labs/brainapi-plugin-search-colbert.git plugins/search-colbert
```

Or:

```bash
./bin/brainapi install search-colbert
```

Restart the API. Encoding needs `torch` and `transformers`. Optional `numpy` speeds up MaxSim (a pure-Python fallback exists). The checkpoint is lazy-loaded on first encode. Device order: MPS → CUDA → CPU.

## Quick start

```bash
curl -X POST "$BRAINAPI_URL/search-colbert/index" \
  -H "Content-Type: application/json" \
  -H "BrainPAT: $BRAINPAT_TOKEN" \
  -d '{"brain_id": "searchbenchsmoke", "limit": 1000}'

curl -X POST "$BRAINAPI_URL/retrieve/search" \
  -H "Content-Type: application/json" \
  -H "BrainPAT: $BRAINPAT_TOKEN" \
  -H "X-Brain-ID: searchbenchsmoke" \
  -d '{
    "query": "navy wool coat",
    "k": 50,
    "channels": ["plugin:colbert"]
  }'
```

You can fuse with core passages (`channels: ["passages", "plugin:colbert"]`). Frozen-head merge of plugin lists into the default passages head is a core `/retrieve/search` behavior — it is **not** the omitted-channels default (`["passages"]` only).

Benchmark harness: `--channels plugin:colbert` after indexing.

## How it retrieves

1. `POST /search-colbert/index` pages text chunks (up to `limit`, max 20 000) and encodes each doc to a list of token vectors (CLS/SEP stripped when possible). Max sequence length **180**.
2. A query is encoded the same way.
3. Score is **MaxSim**: for each query token, take max cosine similarity over document tokens, then sum.
4. Top `k` chunk ids go back to `/retrieve/search`.

The index lives **in process**. Restarting the API clears it. `index_chunks(..., replace=True)` (the HTTP route) resets that brain first.

This is a first-stage plugin, not PLAID-scale serving. Expect it to be slower than BM25/dense on large catalogs.

## Configuration

| Env | Default |
|---|---|
| `SEARCH_COLBERT_MODEL` | `colbert-ir/colbertv2.0` |

Tests can inject `set_encoder(fn)`.

## API

### `GET /search-colbert/health?brain_id=`

```json
{
  "plugin": "search-colbert",
  "channel": "plugin:colbert",
  "model": "colbert-ir/colbertv2.0",
  "loaded": false,
  "error": null,
  "index": { "brain_id": "searchbenchsmoke", "n_docs": 2043 }
}
```

`index` is included only when `brain_id` is passed.

### `POST /search-colbert/index`

```json
{ "brain_id": "searchbenchsmoke", "limit": 1000 }
```

`limit` is `1…20000` (default 1000). Returns `{ brain_id, n_docs }`.

## Layout

```text
search-colbert/
  plugin.yaml
  main.py       # register_search_retriever("colbert", …)
  encode.py     # ColBERT token encoder
  index.py      # MaxSim index + retrieve
  routes.py     # health + index
```

## Publishing

Pushes to `main` publish to the BrainAPI registry via GitHub Actions.

## License

Business Source License 1.1. See [LICENSE](LICENSE).

## Related

- [search-splade](https://github.com/Lumen-Labs/brainapi-plugin-search-splade)
- [search-rerank](https://github.com/Lumen-Labs/brainapi-plugin-search-rerank)
- [BrainAPI](https://github.com/Lumen-Labs/brainapi2)
- `docs/research/18-search-eval-protocol.md` on brainapi2
