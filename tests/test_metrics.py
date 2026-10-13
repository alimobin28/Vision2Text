import pytest

from src.evaluation.metrics import bleu_scores, cider_d, rouge_l, score_all

REFS = [
    ["a red circle on a blue background", "red circle with blue background"],
    ["a green square on a yellow background", "green square with yellow background"],
    ["a purple triangle on an orange background", "purple triangle with orange background"],
    ["a dog runs through the green grass", "dog running on grass"],
]


def test_rouge_l_identical_and_disjoint():
    assert rouge_l("a b c", ["a b c"]) == pytest.approx(1.0)
    assert rouge_l("x y z", ["a b c"]) == 0.0
    assert rouge_l("", ["a b c"]) == 0.0


def test_rouge_l_known_value():
    # hyp "a b c" vs ref "a c": LCS = 2 -> P = 2/3, R = 1, beta = 1.2
    p, r, b2 = 2 / 3, 1.0, 1.2**2
    assert rouge_l("a b c", ["a c"]) == pytest.approx((1 + b2) * p * r / (r + b2 * p))


def test_rouge_l_takes_best_reference():
    assert rouge_l("a b c", ["x y z", "a b c"]) == pytest.approx(1.0)


def test_bleu_perfect_match():
    hyps = [r[0] for r in REFS]
    scores = bleu_scores(hyps, REFS)
    assert set(scores) == {"BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"}
    assert all(v == pytest.approx(1.0) for v in scores.values())


def test_bleu_drops_for_wrong_captions():
    wrong = ["a cat sits on a mat"] * len(REFS)
    assert bleu_scores(wrong, REFS)["BLEU-4"] < 0.1


def test_cider_ranks_exact_above_partial_above_unrelated():
    exact = [r[0] for r in REFS]
    partial = ["a red circle", "a green square", "a purple triangle", "a dog"]
    unrelated = ["cat sleeping sofa"] * 4
    s_exact, s_partial, s_unrelated = (cider_d(h, REFS) for h in (exact, partial, unrelated))
    assert s_exact > s_partial > s_unrelated
    assert s_unrelated == pytest.approx(0.0)
    assert s_exact > 1.0     # CIDEr is x10-scaled; a perfect match is well above 1


def test_cider_edge_cases():
    assert cider_d([], []) == 0.0
    with pytest.raises(ValueError):
        cider_d(["a"], [["a"], ["b"]])


def test_score_all_keys_and_ranges():
    scores = score_all([r[0] for r in REFS], REFS)
    assert set(scores) == {"BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L", "CIDEr-D"}
    assert scores["ROUGE-L"] == pytest.approx(1.0)
