# hwocr: offline handwritten PDF / image → text

A complete local pipeline that turns handwritten PDFs and photos/scans into clean text. It keeps
**line breaks and paragraphs**, then saves `.txt`, `.md` and structured `.json` (pages → paragraphs → lines).
Everything runs on your own machine with an open-weight vision-language model (VLM). No data leaves the computer.

```
PDF / JPG / PNG / TIFF / HEIC
   │  ingest      PyMuPDF renders PDF pages at 200-300 dpi; EXIF rotation fixed; multi-page TIFF
   ▼
   │  preprocess  deskew (±5°), resize to the model's sweet spot, optional contrast boost/denoise
   ▼              (no binarisation: VLMs read natural images better than thresholded ones)
   │  transcribe  local VLM with a strict "copy exactly, keep lines & paragraphs" prompt
   ▼              → loop/truncation detection → retry → fall back to reading the page in strips
   │  refine      (optional) model proof-reads its own draft against the image
   ▼
   │  merge       strips stitched back together, duplicate overlap lines removed
   ▼
   output/<name>.txt  .md  .json         + `eval` / `bench` to measure accuracy on your pages
```

---

## 1. Why your old OCR got 20-30%, and why this approach is different

Classic OCR engines (Tesseract and similar) were built for **printed** text. They first segment the page into lines and
characters, then recognise each one. Handwriting breaks both steps: lines slope and touch, letters join,
and spacing is irregular. The result is garbled characters and lost line/paragraph structure, which is what you saw.

Vision-language models read the whole page at once, like a person. They use context ("th_ quick" → "the") to decode
ambiguous letters, and they can write the output with the original line and paragraph breaks.
That is the change that moves accuracy from 20-30% to the 80-95% range on typical handwriting.

## 2. Research summary: which model?

I surveyed the 2025-2026 benchmarks before choosing a default. Key findings:

* **Document-OCR leaderboards are dominated by printed text.** On OmniDocBench v1.6, small specialists
  lead: PaddleOCR-VL-1.6 (96.3), MinerU2.5-Pro (95.8), GLM-OCR (95.2). These are excellent for printed and mixed documents.
  A high score there does **not** guarantee handwriting performance.
* **On handwriting-specific benchmarks, general VLMs from the Qwen family lead among open models.**
  * *OmniHandwritingOCR* (2026): **Qwen3-VL-8B** has the strongest aggregate score of all evaluated systems.
    The authors also stress that handwriting OCR is "far from solved".
  * *Handwritten Swiss signature lists* (2026): **Qwen3-VL-8B** performed best of the models tested.
  * Independent local test on handwritten journal pages: `qwen3-vl:8b` reached **~5% word error rate (≈95% word accuracy)**.
    Claude Opus reached ~3%. The tester also noted the model could get stuck "thinking" with a loose prompt,
    and a strict OCR prompt fixed it. This pipeline uses a strict prompt and disables thinking.
  * *WildHandBench* (2026, 18 models, includes Qwen3-VL, Qwen3.5, GLM-OCR, PaddleOCR-VL, MinerU, dots, DeepSeek-OCR2):
    the best model scores 71.9% vs 77.1% for humans on "in-the-wild" handwriting. Messy real-world writing is hard for everyone.
* **Qwen3.5** (Feb 2026, Apache-2.0, 0.8B-397B sizes) is the newer generation. Practitioners report **Qwen3.5-9B** is the
  sweet spot for OCR, and that sizes below ~4B drift into summarising instead of transcribing. No handwriting-specific
  benchmark numbers yet, so it is the main challenger to test against the default.
* **Chandra 2** (Datalab, Mar 2026) tops olmOCR-bench (85.9) and claims "excellent" handwriting. Note the weights licence:
  free for research, personal use and companies under $2M revenue/funding, with a paid licence above that.

You mentioned Qwen "2.0" working well six months ago. Qwen2/2.5-VL are the predecessors of the Qwen3-VL / Qwen3.5 models below.

### Recommended models

| Model | Params | GPU memory (4-bit) | Use it when | Ollama name* |
|---|---|---|---|---|
| **Qwen3-VL-8B-Instruct** (default) | 8B | ~6-8 GB | Best-evidenced open model for handwriting | `qwen3-vl:8b` |
| Qwen3.5-9B | 9B | ~7-9 GB | Newer generation; benchmark it against the default | `qwen3.5:9b` |
| Qwen3-VL-32B / 30B-A3B | 32B / 30B MoE | ~20-24 GB | You have a 24 GB GPU and need the extra accuracy | `qwen3-vl:32b`, `qwen3-vl:30b` |
| Qwen3-VL-4B | 4B | ~4 GB | Small GPU / laptop; expect lower accuracy | `qwen3-vl:4b` |
| GLM-OCR / PaddleOCR-VL-1.6 | ~0.9B | ~2 GB | Mostly printed forms with some handwriting; very fast | check `ollama search` |
| Chandra 2 | ~9B | ~8 GB | Mixed layouts and tables + handwriting (mind the licence) | serve with vLLM |

\*Ollama tags change; confirm with `ollama search qwen3` or on ollama.com. Any model works with this pipeline;
**use `hwocr bench` to choose by measurement on your own pages** (section 5).

CPU-only works but is slow (expect minutes per page for an 8B model). A GPU with 8 GB+ VRAM is strongly recommended.

## 3. Install

```bash
cd handwriting-ocr
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then pick **one** way to run the model locally.

**A) Ollama (easiest, Windows/macOS/Linux)**
```bash
# install from https://ollama.com, then:
ollama pull qwen3-vl:8b
python -m hwocr doctor          # -> OK: Ollama at http://localhost:11434, model 'qwen3-vl:8b' available
```

**B) vLLM (fastest on an NVIDIA GPU, good for large batches)**
```bash
pip install vllm
vllm serve Qwen/Qwen3-VL-8B-Instruct --max-model-len 16384 --limit-mm-per-prompt '{"image":1}'
python -m hwocr doctor --backend openai --base-url http://localhost:8000/v1 --model Qwen/Qwen3-VL-8B-Instruct
```
llama.cpp's `llama-server` (with the model's `mmproj` file) and LM Studio also work with `--backend openai`.

**C) In-process Hugging Face transformers (no server)**
```bash
pip install -r requirements-transformers.txt
python -m hwocr run notes.pdf --backend transformers --hf-model Qwen/Qwen3-VL-8B-Instruct
```

**Fully offline:** after the one-time model download (`ollama pull` or the first HF run), no internet is needed.
For transformers, set `HF_HUB_OFFLINE=1` to guarantee no network calls.

## 4. Use

```bash
# one file, many files, or whole folders (recursive)
python -m hwocr run scans/letter.pdf photos/ -o output

# useful options
python -m hwocr run notes.pdf --pages 1-5           # only some pages
python -m hwocr run notes.pdf --refine              # 2nd proof-reading pass: slower, usually more accurate
python -m hwocr run notes.pdf --tiling always       # dense / small writing: read the page in strips
python -m hwocr run notes.pdf --contrast            # faint pencil or faded ink
python -m hwocr run notes.pdf --language Urdu --rtl # non-English / right-to-left scripts
python -m hwocr run notes.pdf --context "doctor's prescription"
python -m hwocr run notes.pdf --rotate 90           # sideways scans
python -m hwocr run notes.pdf --debug-images        # save what the model actually sees
python -m hwocr run big_folder/ --workers 4         # parallel requests (vLLM)
```

Output for `letter.pdf`:

* `output/letter.txt`: plain text. Lines are kept, paragraphs are separated by a blank line.
* `output/letter.md`: Markdown with a heading per page; line breaks are preserved when rendered.
* `output/letter.json`: `pages[].paragraphs[].lines[]`, plus per-page timing, strategy and flags.
* `output/.cache/`: per-page results. Re-running skips pages already done with the same settings,
  so an interrupted 300-page job resumes where it stopped. Use `--overwrite` to redo them.

At the end the run lists any pages flagged `truncated` or `degenerate_output`. Check those by hand.

All defaults live in `config.yaml` (every option is documented there). Use `-c myconfig.yaml` for your own.

## 5. Measuring accuracy and reaching ≥80%

"Accuracy" must be measured on **your** documents. Handwriting quality varies more than any benchmark.

1. Pick 10-20 representative pages and type out exactly what is written. Save them as `samples/ground_truth/<name>.txt`,
   with the source files in `samples/input/<name>.pdf|jpg`. Keep line breaks and use blank lines between paragraphs.
2. Compare models:
   ```bash
   python -m hwocr bench --inputs samples/input --gt samples/ground_truth \
       --models qwen3-vl:8b,qwen3.5:9b,qwen3-vl:4b
   ```
   ```
   | model       | char acc | word acc | line F1 | paragraphs exact | s/page |
   ```
   * **char acc** = 1 − CER (character error rate). **word acc** = 1 − WER and is stricter: one wrong letter makes the whole word wrong.
   * **line F1** = how well line breaks match the page. **paragraphs exact** = share of documents with the right paragraph count.
3. Score any output folder: `python -m hwocr eval --pred output --gt samples/ground_truth`.

If you are below 80%, work down this list (re-run `bench` after each step):

| Step | Typical effect |
|---|---|
| `--dpi 300` for PDFs, and `--contrast` for faint writing | helps thin / light strokes |
| `--refine` | fixes misreads and dropped lines; ~2× time |
| `--tiling always --strips 3` | big gain on dense pages of small writing |
| A larger model (Qwen3-VL-32B / 30B-A3B) | generally the biggest single jump if you have the GPU |
| `--language` / `--context` hints | helps with names, domain terms, non-Latin scripts |
| **Fine-tune on your own handwriting** | best option for one writer or one form type |

**Fine-tuning:** if the documents come from a small set of writers or a fixed form, a LoRA fine-tune of Qwen3-VL on
~100-300 of your own transcribed pages usually produces the largest accuracy gain of anything on this list. The
`samples/ground_truth` files you create for benchmarking double as training data. Unsloth and LLaMA-Factory both support
Qwen-VL LoRA on a single consumer GPU. Export the result to Ollama/vLLM and point `--model` at it; nothing else in the
pipeline changes.

## 6. Project layout

```
hwocr/
  ingest.py       PDF/image/TIFF loading, page ranges, folder discovery
  preprocess.py   deskew, resize, contrast, denoise, whitespace-aware strip splitting
  prompts.py      transcription + refinement prompts (the rules that preserve lines/paragraphs)
  backends.py     Ollama, OpenAI-compatible (vLLM/llama.cpp/LM Studio), HF transformers
  postprocess.py  output cleaning, loop detection, strip merging, paragraph parsing
  pipeline.py     orchestration, retries/fallbacks, caching/resume, parallelism
  output.py       txt / md / json writers
  evaluate.py     CER, WER, line F1, paragraph metrics
  cli.py          `run`, `doctor`, `eval`, `bench`
config.yaml       all settings, documented
tests/            34 tests (run: pip install pytest && pytest)
```

## 7. Troubleshooting

* **Output stops halfway down the page.** The page was flagged `truncated` and automatically re-read in strips. If it persists,
  raise `--max-tokens` or use `--tiling always`. On Ollama, keep `num_ctx` ≥ 16384 in `config.yaml`:
  the 4096 default is too small for a page image plus its text.
* **Same line repeated many times.** This is a known VLM failure mode. The pipeline detects it, retries with a higher repetition penalty,
  then falls back to strips. Pages that still fail are listed at the end of the run.
* **Model "explains" or summarises instead of transcribing.** The model is probably too small (<4B). Use 8B+.
* **Sideways pages.** Use `--rotate 90` (or 270). Small tilts are fixed automatically.
