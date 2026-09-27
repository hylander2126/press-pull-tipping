// Links and citation data for the project page, all in one place.
// TODO: add arXiv when available.

export const REPO_URL = "https://github.com/hylander2126/press-pull-tipping";
export const PAPER_URL = "https://stevenmhyland.com/assets/pdfs/ISRR_2026.pdf";
export const ARXIV_URL = "#"; // TODO: e.g. https://arxiv.org/abs/XXXX.XXXXX

export const CODE_URL = REPO_URL;
export const PACKAGE_URL = `${REPO_URL}/tree/main/code`;
export const SANDBOX_URL = `${REPO_URL}/blob/main/code/press_pull_estimator/sim/press_pull_sandbox.py`;
export const ESTIMATOR_URL = `${REPO_URL}/blob/main/code/press_pull_estimator/estimator/wrench_estimator.py`;

export const BIBTEX = `@inproceedings{hyland2026pressandpull,
  title     = {Multi-Modal Non-Prehensile Estimation of Physical Parameters via Press-and-Pull Tipping},
  author    = {Hyland, Steven M. and Xiao, Jing and Onal, Cagdas D.},
  booktitle = {International Symposium on Robotics Research (ISRR)},
  year      = {2026}
}`;

/** Prefix a path in public/ with the deploy base (GitHub Pages serves from /<repo>/). */
export function withBase(path: string): string {
  const base = import.meta.env.BASE_URL;
  return (base.endsWith("/") ? base : base + "/") + path.replace(/^\//, "");
}
