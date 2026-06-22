# Scientific Computing Agent — Design Specification

**Date:** 2026-06-22
**Status:** Approved
**Branch:** `calculate_agent`

## 1. Overview

Transform the generic `agent_learning` platform into a **chemistry & physics computational agent** targeting upper-level undergraduates and first-year graduate students.

### Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LLM backend | Ollama (pure local) | No API cost, student-friendly, server-side deployment |
| Deployment | Server-side, browser access | One GPU server serves all students in a lab/classroom |
| Scope | Upper undergrad + 1st-year grad | Quantum mechanics, stat thermo, spectroscopy, EM, etc. |
| Knowledge base | Public authoritative sources + citations | IUPAC, NIST, Wikipedia, OpenStax textbooks |
| Architecture approach | Keep MCP skeleton, inject domain tools | Maximize reuse of proven infrastructure |

### Principles

1. **Each phase has a test gate** — no bulk testing at the end
2. **Backward compatible** — generic agent features not broken during migration
3. **Accuracy over speed** — sympy symbolic computation + verify chain + cited knowledge

---

## 2. Architecture

```
                    ┌── Browser (H5 Frontend) ──┐
                    │  chat.js + KaTeX render   │
                    └──────────┬────────────────┘
                               │ HTTP/SSE/WS
                    ┌──────────▼────────────────┐
                    │   FastAPI (server.py)      │
                    │   ┌─────────────────────┐  │
                    │   │  ReAct Engine        │  │
                    │   │  + verify chain (new)│  │
                    │   └──┬──────────────────┘  │
                    │      │ ToolRegistry         │
                    │   ┌──┴───────────────────┐  │
                    │   │  Local: calculator     │  │
                    │   │  + timer_set           │  │
                    │   └───────────────────────┘  │
                    │   ┌──┴───────────────────┐  │
                    │   │  MCP Servers (stdio)   │  │
                    │   │  chemistry_server (new)│  │
                    │   │  physics_server (new)  │  │
                    │   │  verify_server (new)   │  │
                    │   │  legacy 5 servers      │  │
                    │   └───────────────────────┘  │
                    └──────────────────────────────┘
                               │
                    ┌──────────▼────────────────┐
                    │  Ollama (local LLM)        │
                    │  Model: qwen3:14b or equiv │
                    └──────────────────────────────┘
                               │
                    ┌──────────▼────────────────┐
                    │  ChromaDB                  │
                    │  science_kb (new collection)│
                    │  rag_documents (existing)  │
                    │  semantic_all (existing)   │
                    └──────────────────────────────┘
```

### What stays

- FastAPI + WebSocket entry points
- MCP protocol layer (ToolRegistry, MCPClientManager, ToolProtocol)
- ReAct + Plan-Solve dual engine architecture
- Shell/filesystem/document/memory MCP Servers (legacy tools coexist)
- Three-layer memory system
- Evaluation system
- H5 frontend (Apple HIG)

### What changes

| File | Change | Phase |
|------|--------|-------|
| `backend/llm_client.py` | Rewrite: Ollama adapter | 1a |
| `backend/agent/tools.py` | Rewrite: calculator → sympy engine | 1b |
| `web/static/js/chat.js` | Add: KaTeX auto-render | 1c |
| `backend/protocols/mcp/servers/chemistry_server.py` | New | 2a |
| `backend/protocols/mcp/servers/physics_server.py` | New | 2b |
| `backend/memory/science_kb_ingest.py` | New | 2c |
| `backend/protocols/mcp/servers/verify_server.py` | New | 3b |
| `backend/agent/engine.py` | Modify: verify chain hook | 3c |
| `backend/memory/embedding.py` | Modify: formula-aware chunking | 3a |
| `requirements.txt` | Add: sympy, scipy, pint, mendeleev, katex | 1 |

---

## 3. Phase Details

### Phase 1a — Ollama Client (1-2 days)

**File:** `backend/llm_client.py`

Replace DeepSeek API with Ollama OpenAI-compatible endpoint.

```python
# Key changes:
LLM_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL_ID = os.getenv("OLLAMA_MODEL", "qwen3:14b")

class LLMClient:
    # Keep: chat(), chat_stream() — same signatures
    # Add:  list_models() — query available Ollama models
    # Drop: get_cache_info() — Ollama doesn't support prompt caching
```

**Environment (.env):**
```bash
OLLAMA_BASE_URL="http://localhost:11434/v1"
OLLAMA_MODEL="qwen3:14b"
```

**Test gate:**
- Non-streaming call returns valid response
- Streaming SSE events complete without gaps
- Model switch via `.env` works on restart
- Ollama unreachable → graceful error, not crash
- `estimate_tokens()` accuracy < 30% deviation for mixed CN/EN input

---

### Phase 1b — sympy Calculation Engine (2-3 days)

**File:** `backend/agent/tools.py` → `calculator()` handler

Replace `eval()`-based safe calculator with sympy symbolic engine.

**Capabilities:**
- Symbolic: `diff`, `integrate`, `solve`, `dsolve`, `limit`, `series`
- Matrix: `eigenvals`, `diagonalize`, `det`, `inv`
- Numeric: `N()` / `evalf(n)` for arbitrary precision
- Units: pint integration for dimensional analysis and conversion

**Input format (extended):**
```
calc:diff(x**3 + sin(x), x)          → symbol
calc:integrate(x**2, (x, 0, 1))     → definite integral
calc:solve(x**2 - 4, x)             → equation
calc:eval(pi, 50)                   → high-precision numeric
calc:unit(5*m/s, km/h)              → unit conversion
```

**Test gate:**
- Code injection blocked (`__import__`, `eval`, `exec` rejected)
- `diff(x**3 + sin(x), x)` → `3*x**2 + cos(x)`
- `integrate(x**2, (x, 0, 1))` → `1/3`
- `solve(x**2 - 4, x)` → `[-2, 2]`
- `5 * ureg.meter / ureg.second → km/h` → `18.0 km/h`
- `pi.evalf(50)` → 50 significant digits
- Division by zero → friendly error
- Oversized input → no OOM

---

### Phase 1c — LaTeX Frontend Rendering (0.5 day)

**File:** `web/static/js/chat.js`, `web/static/index.html`

Add KaTeX auto-render to the chat message pipeline.

```html
<!-- index.html: add KaTeX CDN -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16/dist/contrib/auto-render.min.js"></script>
```

```javascript
// chat.js: render after each message
renderMathInElement(messageEl, {
    delimiters: [
        {left: '$$', right: '$$', display: true},
        {left: '$', right: '$', display: false},
        {left: '\\(', right: '\\)', display: false},
        {left: '\\[', right: '\\]', display: true}
    ]
});
```

**Test gate:**
- Inline `$E=mc^2$` renders as inline LaTeX
- Block `$$\int_0^\infty f(x)dx$$` renders as display math
- Chinese + formula mixed content wraps correctly
- Chrome, Firefox, Edge all render

---

### Phase 2a — Chemistry MCP Server (2-3 days)

**File:** `backend/protocols/mcp/servers/chemistry_server.py`

Stdio JSON-RPC MCP Server. Dependencies: `sympy`, `mendeleev`.

**Tools:**

| Tool | Function | Input example |
|------|----------|---------------|
| `balance_equation` | Stoichiometric balancing | `CH4 + O2 -> CO2 + H2O` |
| `thermo_calc` | ΔH/ΔG/ΔS calculations | Reaction, temperature, standard enthalpies |
| `equilibrium` | Equilibrium constants | Reaction, initial concentrations |
| `kinetics` | Rate laws, Arrhenius | Reaction order, k, T, Ea |
| `solution_chem` | pH, buffers, titrations | Acid/base, concentration, volume |
| `electrochem` | Nernst, cell potential | Half-reactions, concentrations, T |
| `element_lookup` | Element/compound properties | Symbol, name, or formula |

**Test gate:**
- `Fe + Cl2 -> FeCl3` → coefficients [2, 3, 2]
- `H2SO4` molar mass → 98.079 g/mol
- 0.1M HCl → pH = 1.00; 0.1M HAc → pH ≈ 2.87
- `Cu²⁺/Cu` standard potential → +0.34 V
- Incomplete combustion → reasonable product set, not error
- Invalid formula → friendly "unrecognized" message

---

### Phase 2b — Physics MCP Server (2-3 days)

**File:** `backend/protocols/mcp/servers/physics_server.py`

Stdio JSON-RPC MCP Server. Dependencies: `sympy`, `scipy`, `pint`.

**Tools:**

| Tool | Function |
|------|----------|
| `mechanics` | Kinematics, Newton's laws, energy, momentum |
| `electromagnetism` | Coulomb, Biot-Savart, Maxwell equations |
| `thermodynamics` | Carnot cycle, entropy, free energy |
| `quantum` | 1D infinite well, harmonic oscillator, H-atom eigenvalues. **Only analytically solvable models.** Multi-electron systems (He, Li, ...) must return "not analytically solvable — suggest numerical methods (HF/DFT)" instead of a wrong number |
| `optics` | Lens equation, interference, diffraction |
| `error_propagation` | Error synthesis for ±/*/log/exp operations |

**Test gate:**
- Kinematics: `s = ut + ½at²` → exact value for given params
- Coulomb: two 1μC charges at 1m → F = 8.99×10⁻³ N
- Harmonic oscillator: m=1, k=100 → ω = 10 rad/s
- Infinite well: L=1nm ground state → reasonable energy
- Unit conversion: `10 m/s → 36 km/h` via pint
- Dimensional analysis: force ≠ mass → pint detects error
- Quantum boundary: `"helium atom ground state energy"` → refuses with "not analytically solvable" message, not a guessed number

---

### Phase 2c — Scientific Knowledge Base (3-5 days)

**New file:** `backend/memory/science_kb_ingest.py`
**New collection:** `science_kb` (separate from `rag_documents`)

**Data sources:**

| Source | Content type | Citation format |
|--------|-------------|-----------------|
| IUPAC Gold Book | Terms, definitions, standard values | `IUPAC Gold Book, <id>` |
| NIST Chemistry WebBook | Thermochemical data, spectra | `NIST WebBook, doi:10.18434/T4D303` |
| Wikipedia Chemistry/Physics | Concepts, formulas | `Wikipedia, <page>, rev.<date>` |
| CRC Handbook (public) | Physical constants, math tables | `CRC Handbook, 104th ed., §<n>` |
| OpenStax textbooks | Systematic knowledge | `OpenStax, <book>, Ch.<n>` |

**Formula-aware chunking:**
```
Detect $...$, $$...$$, \[...\] boundaries
→ Treat formulas as atomic units (never split mid-formula)
→ Each chunk metadata: {source_url, doi, subject_tags, formulas[], last_updated}
```

**Embedding model selection:**
```
Default: BGE-small-zh-v1.5 (512-dim, existing in project)
Risk: BGE-small-zh trained primarily on Chinese text.
      IUPAC/NIST/Wikipedia-English may have lower recall.
Fallback: bge-m3 (multilingual, 1024-dim) or intfloat/e5-small-v2

Decision gate in test: run English-vs-Chinese query comparison.
If BGE-small-zh English recall < 70% of Chinese recall → switch to bge-m3.
```

**Ingestion pipeline:**
```
Source URL / file
  → fetch/parse
  → formula-aware split
  → embed (BGE-small-zh-v1.5 or bge-m3, TBD by test)
  → upsert to science_kb collection
  → log {chunk_count, source, timestamp}
```

**rag_search collection routing (bridge to science_kb):**
```
rag_search tool adds optional `collection` parameter:
  - "all" (default) → search rag_documents + science_kb, merge by relevance
  - "science_kb" → search science_kb only
  - "rag_documents" → search rag_documents only

This ensures science_kb is usable immediately after Phase 2c ingest,
not delayed until Phase 3b.
```

**Test gate:**
- 100 items ingested → `science_kb.count() == 100`
- `"hydrogen ground state energy"` → top-3 hits relevant
- `"hydrogen Lyman series wavelength"` (English query, English docs) → relevant chunks returned. If BGE-small-zh recall < 70% of equivalent Chinese query, switch to bge-m3
- No orphaned `$` or `$$` in any chunk
- Retrieved chunks have non-empty `source` field
- Duplicate URL → skipped, not duplicated
- `rag_search("electron affinity", collection="science_kb")` → hits science_kb only, not rag_documents

---

### Phase 3a — RAG Metadata Enhancement (2 days)

**Files:** `backend/memory/embedding.py`, ingestion pipeline

**Enhancements:**
1. **Auto-tagging**: LLM classifies each chunk → `["thermodynamics", "quantum", ...]`
2. **Formula index**: Extract all formulas from chunk → `["e^{iπ}+1=0", "E=mc²"]`
3. **Confidence tiers**: `peer-reviewed` > `textbook` > `wikipedia` > `other`
4. **Rank by confidence**: higher-confidence sources rank first for same relevance score

**Test gate:**
- Random 20 tags → human audit ≥ 85% correct
- Search `E=mc²` → hits mass-energy chunks
- `peer-reviewed` sources sort above `wikipedia` for equal relevance
- Existing `rag_documents` collection unaffected

---

### Phase 3b — Verify MCP Server (2 days)

**File:** `backend/protocols/mcp/servers/verify_server.py`

Post-hoc verification. Does NOT block the main response — results appended as a collapsible verification report.

**Verification dimensions:**

| Dimension | Method | Example |
|-----------|--------|---------|
| Dimensional analysis | pint parse → check expected dimension | Speed result with kg units → flag |
| Order-of-magnitude | Compare against known constants | c = 300 m/s → flag (off by 10⁶) |
| Symbolic back-substitution | sympy substitute result into original equation | x=2 into x²-4 → 0 ✓ |
| Knowledge cross-check | RAG search answer claims vs knowledge base | Specific heat claim vs NIST value |
| Dual-path | Solve same problem two ways | Energy method vs force method |

**Test gate:**
- Correct dimension: `F=10N, a=2m/s² → m=5kg` → passes
- Wrong dimension: `F = 10 kg` → flagged suspicious
- Back-sub: `solve(x²-5x+6=0)` roots → substitution yields 0
- Non-blocking: verification failure shows warning, answer already delivered
- Per-claim latency ≤ 5s (single claim: classify → verify one dimension)
- Total report latency ≤ 15s (all claims verified in parallel via asyncio.gather)

---

### Phase 3c — Reasoning Engine Verify Chain (1-2 days)

**File:** `backend/agent/engine.py`

Add `_verify_claims()` step after ReAct loop concludes:

```
ReAct round 3 → LLM produces answer
  │
  ├── Extract claims from answer text
  ├── Classify each claim:
  │     ├── Calculation → Verify Server (sympy back-sub)
  │     ├── Factual statement → RAG cross-check
  │     └── Reasoning chain → LLM self-review
  │
  └── Generate verification report → append to answer
```

**Changes to engine.py:**
- New method: `_extract_claims(answer: str) -> list[Claim]`
- New method: `_verify_claims(claims: list[Claim]) -> VerificationReport`
- Hook into existing post-reflection step
- Configurable: `VERIFY_ENABLED=true/false` in `.env`

**Verification report format:**
```markdown
### Verification Report
[✓] Claim 1: "E = 2.18×10⁻¹⁸ J" — matches NIST value (2.179×10⁻¹⁸, 0.1% dev)
[✓] Claim 2: "λ = hc/E = 91.2 nm" — back-substitution verified
[!] Claim 3: "electron mass = 9.11×10⁻³¹ kg" — standard value (not verified)
```

**Test gate:**
- `"speed of light 2.99×10⁸ m/s, photon energy E=hf"` → extracts 2 claims
- Erroneous calculation → `[!]` flag in report
- Report renders as collapsible section in frontend
- Verification adds ≤ 5s total latency

---

## 4. Dependency Graph

```
Phase 1a (Ollama)    Phase 1b (sympy)    Phase 1c (LaTeX)
     │                     │                    │
     └─────────┬───────────┘                    │
               │                                │
      Phase 2a (Chemistry)  Phase 2b (Physics)  │
               │                    │           │
               └─────────┬──────────┘           │
                         │                      │
                 Phase 2c (Sci KB)              │
                         │                      │
                 Phase 3a (RAG enhance)          │
                         │                      │
                 Phase 3b (Verify Server)        │
                         │                      │
                 Phase 3c (Verify chain) ←───────┘
```

**Parallelism opportunities:**
- Phase 1a, 1b, 1c can run concurrently
- Phase 2a, 2b can run concurrently (shared sympy dependency)
- Phase 2c can start early (no dependency on 2a/2b)

## 5. Effort Estimates

| Phase | Task | Work | Calendar (parallel) |
|-------|------|------|---------------------|
| 1a | Ollama client | 1-2d | ═══╗ |
| 1b | sympy engine | 2-3d | ═══╣ 3d |
| 1c | LaTeX render | 0.5d | ═══╝ |
| 2a | Chemistry Server | 2-3d | ═══╗ |
| 2b | Physics Server | 2-3d | ═══╣ 4d |
| 2c | Science KB | 3-5d | ═══╝ |
| 3a | RAG enhance | 2d |      |
| 3b | Verify Server | 2d | ═══╗ 3d |
| 3c | Verify chain | 1-2d | ═══╝ |
| **Total** | | **13-20d** | **10d** |

---

## 6. Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Ollama model insufficient for scientific reasoning | High | Test with multiple models (Qwen, DeepSeek-Coder, Llama); document recommended models |
| sympy performance on large symbolic expressions | Medium | Timeout wrapper (30s); fallback to numeric approximation |
| Knowledge base source URLs changing/breaking | Medium | Version-pin Wikipedia revisions; cache fetched content |
| RAG retrieval poor for formula-heavy queries | Medium | Formula index as secondary retrieval path (Phase 3a) |
| Browser KaTeX rendering on slow clients | Low | SSR fallback option (render server-side as SVG) |
| MCP Server startup overhead (4 new servers = 4 more Python processes) | Low | Consider merging chemistry+physics into one `science_server.py` if memory becomes an issue |

---

## 7. Open Questions (deferred)

1. **Recommended Ollama models**: Which model works best for scientific reasoning? (Qwen3-14B vs DeepSeek-Coder-V2 vs Llama-4?)
   → *Decision deferred to Phase 1a testing*
2. **Chemistry structure diagrams**: Should molecular structures be rendered (RDKit → SVG) or text-only?
   → *Deferred to post-Phase-2 feature request*
3. **Multi-language support**: English-only or Chinese + English?
   → *Current scope: Chinese interface + bilingual knowledge base; BGE-small-zh handles both*
