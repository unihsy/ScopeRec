import argparse
import ast
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pyarrow.parquet as parquet
import torch

from scoperec.experiment import write_json
from scoperec.official_model import ReleasedTIGER
from scoperec.official_tokens import encode_requests, metric_rows, released_codes, token_layout
from scoperec.prepare import file_fingerprint
from scoperec.train import setup_torch

REFERENCE = Path("artifacts/official_benchmark/upstream")


def reference_method(path, class_name, method_name):
    tree = ast.parse(path.read_text())
    definition = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in definition.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    namespace = {"torch": torch, "np": np}
    from collections import defaultdict

    namespace["defaultdict"] = defaultdict
    executable = ast.Module(body=[method], type_ignores=[])
    exec(compile(executable, str(path), "exec"), namespace)
    return namespace[method_name]


def parity(device):
    setup_torch(2024, device)
    torch.set_float32_matmul_precision("highest")
    tokenizer_path = REFERENCE / "genrec/models/TIGER/tokenizer.py"
    evaluator_path = REFERENCE / "genrec/evaluator.py"
    model_path = REFERENCE / "genrec/models/TIGER/model.py"
    layout = token_layout()
    codes = np.load("artifacts/official_benchmark/sid/codes.npy")
    catalog = parquet.read_table("data/official_software/catalog.parquet")[
        "parent_asin"
    ].to_pylist()
    raw_semantic = codes[:, :3] - np.arange(3)[None, :] * 256 - 1
    tokenizer = SimpleNamespace(
        id2item=["[PAD]", *catalog],
        codebook_sizes=[256] * 4,
        n_digit=4,
        log=lambda *args: None,
        base_user_token=1025,
        n_user_tokens=1,
        eos_token=1026,
        padding_token=0,
        config={"max_item_seq_len": 20},
        max_token_seq_len=82,
    )
    for name in (
        "_extend_semantic_ids",
        "_sem_ids_to_tokens",
        "_token_single_user",
        "_token_single_item",
        "_tokenize_once",
    ):
        setattr(
            tokenizer,
            name,
            MethodType(reference_method(tokenizer_path, "TIGERTokenizer", name), tokenizer),
        )
    tokenizer.item2tokens = tokenizer._sem_ids_to_tokens(
        tokenizer._extend_semantic_ids(raw_semantic)
    )
    expected_codes = np.asarray([tokenizer.item2tokens[item] for item in catalog])
    np.testing.assert_array_equal(expected_codes, codes)
    np.testing.assert_array_equal(released_codes(raw_semantic), codes)
    archive = np.load("data/official_software/valid.npz")
    requests = {name: archive[name][:8] for name in archive.files}
    inputs, labels = encode_requests(requests, codes, layout)
    tokenizer.user2id = {str(user): int(user) for user in requests["user"]}
    for row in range(len(inputs)):
        sequence = [catalog[item] for item in requests["history"][row] if item >= 0]
        sequence.append(catalog[requests["target"][row]])
        expected_input, expected_mask, expected_label = tokenizer._tokenize_once(
            {
                "user": str(requests["user"][row]),
                "item_seq": sequence,
            }
        )
        np.testing.assert_array_equal(expected_input, inputs[row])
        np.testing.assert_array_equal(expected_mask, inputs[row] != 0)
        np.testing.assert_array_equal(expected_label, labels[row])
    model = (
        ReleasedTIGER(layout, d_model=16, d_ff=32, num_layers=2, num_heads=2, d_kv=8)
        .to(device)
        .eval()
    )
    reference = SimpleNamespace(t5=model.t5, device=torch.device(device))
    for name in ("prepare_beam_search_inputs", "beam_search_step", "beam_search_ori"):
        setattr(reference, name, MethodType(reference_method(model_path, "TIGER", name), reference))
    batch = torch.as_tensor(inputs[:2], device=device)
    expected = reference.beam_search_ori(
        batch, batch.ne(0), max_length=6, num_beams=50, num_return_sequences=50
    )
    actual = model.generate(batch, beam_size=50)
    np.testing.assert_array_equal(
        expected[:, 1:].cpu().numpy().reshape(2, 50, 5), actual["sequences"].cpu().numpy()
    )
    evaluator = SimpleNamespace(maxk=50, eos_token=1026, config={"topk": [10, 20, 50]})
    for name in ("calculate_pos_index_fine", "ndcg_at_k", "recall_at_k", "calculate_metrics"):
        setattr(
            evaluator,
            name,
            MethodType(reference_method(evaluator_path, "Evaluator", name), evaluator),
        )
    predicted = actual["codes"].cpu().numpy().copy()
    predicted[0, 3, :3] = labels[0, :3]
    predicted[0, 3, 3] = (labels[0, 3] + 1) % 1027
    predicted[1, 30] = labels[1, :4]
    expected_metrics = evaluator.calculate_metrics(
        torch.tensor(predicted), torch.tensor(labels[:2]), []
    )
    actual_metrics = metric_rows(predicted, labels[:2], 3)
    for cutoff in (10, 20, 50):
        np.testing.assert_allclose(
            actual_metrics[f"ndcg@{cutoff}"],
            expected_metrics[f"ndcg@{cutoff}"].numpy(),
            rtol=1e-6,
            atol=1e-7,
        )
    report = {
        "passed": True,
        "upstream_revision": "e6878d9c7c6e57479e840ccb8c045b11a2bd69b5",
        "sources": [
            file_fingerprint(path) for path in (tokenizer_path, model_path, evaluator_path)
        ],
        "all_item_token_codes_identical": len(codes),
        "tokenized_requests_identical": len(inputs),
        "unmasked_beam_sequences_identical": True,
        "beam_width": 50,
        "decode_steps": 5,
        "released_prefix_ndcg_matches": True,
        "device": device,
        "method": "Executed unchanged official methods via AST with local data/model adapters",
        "caveat": "Numerical functional parity, not a reproduction of pretrained checkpoints",
    }
    write_json(Path("reports/official_benchmark/parity.json"), report)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Compare released tokenizer, Beam Search and evaluator."
    )
    parser.add_argument("--device", default="cuda:3")
    arguments = parser.parse_args()
    print(json.dumps(parity(arguments.device), indent=2))


if __name__ == "__main__":
    main()
