"""
Caption metrics: BLEU-1..4, ROUGE-L, CIDEr-D (and optionally METEOR).

CIDEr-D and ROUGE-L are small, so they are implemented here following the
reference ``coco-caption`` definitions instead of depending on
``pycocoevalcap`` (which needs Java and ships large jars). Inputs are expected
to be already normalised (lower-case, no punctuation — i.e. the output of
``clean_caption``), which is what the PTB tokenizer does in the COCO pipeline.

Numbers from this module are close to, but not guaranteed bit-identical with,
the official coco-caption toolkit; report them as "our implementation".
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence

from torchmetrics.text import BLEUScore

# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence (O(len(a)*len(b)) DP)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(hypothesis: str, references: list[str], beta: float = 1.2) -> float:
    """ROUGE-L F-score of one hypothesis against several references (coco-caption)."""
    hyp = hypothesis.split()
    best_p = best_r = 0.0
    for ref in references:
        ref_t = ref.split()
        lcs = _lcs_length(hyp, ref_t)
        if lcs == 0:
            continue
        best_p = max(best_p, lcs / len(hyp))
        best_r = max(best_r, lcs / len(ref_t))
    if best_p == 0 or best_r == 0:
        return 0.0
    return ((1 + beta**2) * best_p * best_r) / (best_r + beta**2 * best_p)


# ---------------------------------------------------------------------------
# CIDEr-D
# ---------------------------------------------------------------------------

def _ngram_counts(sentence: str, n_max: int = 4) -> Counter:
    words = sentence.split()
    counts: Counter = Counter()
    for n in range(1, n_max + 1):
        for i in range(len(words) - n + 1):
            counts[tuple(words[i : i + n])] += 1
    return counts


def cider_d(
    hypotheses: list[str],
    references: list[list[str]],
    n_max: int = 4,
    sigma: float = 6.0,
) -> float:
    """Corpus CIDEr-D (x10, as reported in the literature).

    Document frequencies are computed over the reference sets of the evaluated
    corpus itself, exactly like coco-caption does.
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must be the same length")
    if not hypotheses:
        return 0.0

    ref_counts = [[_ngram_counts(r, n_max) for r in refs] for refs in references]

    df: defaultdict = defaultdict(float)
    for refs in ref_counts:
        for ngram in {g for counts in refs for g in counts}:
            df[ngram] += 1.0
    log_n_docs = math.log(float(len(references)))

    def vectorise(counts: Counter) -> tuple[list[dict], list[float], int]:
        vec: list[dict] = [defaultdict(float) for _ in range(n_max)]
        norm = [0.0] * n_max
        length = 0
        for ngram, tf in counts.items():
            n = len(ngram) - 1
            idf = log_n_docs - math.log(max(1.0, df[ngram]))
            vec[n][ngram] = float(tf) * idf
            norm[n] += vec[n][ngram] ** 2
            if n == 1:                       # bigram count, as in coco-caption
                length += tf
        return vec, [math.sqrt(x) for x in norm], length

    def similarity(hyp, ref) -> list[float]:
        (vh, nh, lh), (vr, nr, lr) = hyp, ref
        penalty = math.exp(-((lh - lr) ** 2) / (2 * sigma**2))
        out = []
        for n in range(n_max):
            val = sum(min(w, vr[n][g]) * vr[n][g] for g, w in vh[n].items())
            if nh[n] != 0 and nr[n] != 0:
                val /= nh[n] * nr[n]
            out.append(val * penalty)
        return out

    total = 0.0
    for hyp_sentence, refs in zip(hypotheses, ref_counts, strict=True):
        hyp_vec = vectorise(_ngram_counts(hyp_sentence, n_max))
        acc = [0.0] * n_max
        for ref_c in refs:
            for n, s in enumerate(similarity(hyp_vec, vectorise(ref_c))):
                acc[n] += s
        total += (sum(acc) / n_max) / len(refs) * 10.0
    return total / len(hypotheses)


# ---------------------------------------------------------------------------
# BLEU / METEOR
# ---------------------------------------------------------------------------

def bleu_scores(hypotheses: list[str], references: list[list[str]]) -> dict[str, float]:
    """Cumulative BLEU-1..4 with uniform n-gram weights (corpus level)."""
    out = {}
    for n in range(1, 5):
        metric = BLEUScore(n_gram=n)
        out[f"BLEU-{n}"] = float(metric(hypotheses, references))
    return out


def meteor_score_nltk(hypotheses: list[str], references: list[list[str]]) -> float | None:
    """Mean METEOR via NLTK, or None when NLTK/WordNet is unavailable.

    NLTK's METEOR is not the official Java METEOR 1.5 used in papers; its values
    differ, so label it accordingly. Needs ``nltk.download("wordnet")`` once.
    """
    try:
        from nltk.translate.meteor_score import meteor_score
        scores = [
            meteor_score([r.split() for r in refs], hyp.split())
            for hyp, refs in zip(hypotheses, references, strict=True)
        ]
    except (ImportError, LookupError):
        return None
    return sum(scores) / len(scores) if scores else 0.0


def score_all(
    hypotheses: list[str], references: list[list[str]], meteor: bool = False
) -> dict[str, float]:
    """Every metric in one dict, ready to dump as JSON."""
    scores = bleu_scores(hypotheses, references)
    scores["ROUGE-L"] = sum(
        rouge_l(h, r) for h, r in zip(hypotheses, references, strict=True)
    ) / len(hypotheses)
    scores["CIDEr-D"] = cider_d(hypotheses, references)
    if meteor:
        m = meteor_score_nltk(hypotheses, references)
        if m is not None:
            scores["METEOR (nltk)"] = m
    return scores
