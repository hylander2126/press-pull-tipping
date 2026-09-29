# Press-and-Pull Tipping · ISRR 2026

Project page and standalone code for

> **Multi-Modal Non-Prehensile Estimation of Physical Parameters via Press-and-Pull Tipping**
> Steven M. Hyland, Jing Xiao, Cagdas D. Onal, Worcester Polytechnic Institute
> International Symposium on Robotics Research (ISRR) 2026

This repo holds two independent pieces:

| | What | Where |
|---|---|---|
| 🌐 | Project page (Astro + Tailwind + MDX, from [academic-project-astro-template](https://github.com/RomanHauksson/academic-project-astro-template)) | repo root: `src/`, `public/` |
| 🐍 | `press_pull_estimator`: hardware-free Python estimator and all 40 recorded trials | [`code/`](code/README.md) |

## Run the estimator

```bash
cd code
pip install -e .
python run_offline_eval.py --object flashlight   # recovered (m, z_c, mu_t) with error margins
python run_offline_eval.py --object all -q       # reproduces Table 2 of the paper
```

See [`code/README.md`](code/README.md) for details.

## Run the project page

Requires Node ≥ 22.

```bash
npm install
npm run dev        # http://localhost:4321
npm run build      # astro check + static build into dist/
```

Page layout:

- `src/paper.mdx`: all page content, rendered by `src/pages/index.astro`
- `src/site.config.ts`: **paper, arXiv, and repo links and the BibTeX** (fill in the TODOs before publishing)
- `src/components/`: `DemoShowcase`, `EvidenceStrip`, `VideoCard`, `PipelineDiagram`, `ObjectExplorer` (tabs), `WrenchPlot` (Plotly), `BibTeX`
- `src/data/*.json`: results and plot data, **generated** by `code/tools/export_web_data.py` from the released trials, so the page's numbers can't drift from the code
- `public/videos/`: trial clips, shove-task highlight, research overview, and forward-tipping comparison
- `public/posters/`: video stills, shown before playback (including with reduced motion)
- `src/assets/social-preview.svg`: editable social-card source; `public/social-preview.jpg` is the rendered sharing image

The overview is the supplied `Multimodal-downscaled.mp4` (1:50). The 17-second shove highlight
joins its 0:08–0:17 and 1:38–1:46 excerpts, preserving the embedded labels and timing.
The failure comparison is the supplied `pushing_failure.mp4`, labeled 3× in the source.
The clearer flashlight comparison uses 1:02–1:07 of the overview, labeled 5×; the
trial-2 clips beside the quantitative results remain unchanged.
The physical trial players offer edited-speed and approximate real-time playback; slowing a
sped-up export does not restore omitted frames. Overview playback starts only on request.

The website and Python package deliberately share one repository. The header's Code link
points to its root; detailed estimator links point into `code/`. Configure the
Git remote to match `REPO_URL` in `src/site.config.ts` before publishing.

## Deploy

`.github/workflows/astro.yml` builds and deploys to GitHub Pages on every push to `main`. In the repo
settings, set **Pages → Source** to **GitHub Actions**. All asset paths go through the Pages base path
(`/<repo>/`).
