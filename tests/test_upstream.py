from scoperec.upstream import BENCHMARK_FILES, FILES


def test_benchmark_reference_covers_controlling_paths_without_changing_edit_reference():
    assert len(BENCHMARK_FILES) == len(set(BENCHMARK_FILES))
    assert set(FILES).issubset(BENCHMARK_FILES)
    assert "genrec/datasets/AmazonReviews2023/dataset.py" in BENCHMARK_FILES
    assert "genrec/models/TIGER/tokenizer.py" in BENCHMARK_FILES
    assert "genrec/trainer.py" in BENCHMARK_FILES
    assert len(FILES) == 7