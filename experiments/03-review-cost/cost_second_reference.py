"""Recover the second held-out worker's reference on the same frozen items."""
import argparse
import json
from pathlib import Path

import numpy as np

from blind_rate import datasets
from corpus_sweep_inputs import key
from fresh_votes import mbic, polite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("votes", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    protocol = json.loads(Path(__file__).with_name("review_cost_protocol.json").read_text())
    registry = {"GoEmotions": "go", "Wikipedia-Attack": "wiki",
                "D3code": "d3", "Unhealthy-Comments": "ucc"}
    payload = {}
    for name in protocol["second_reference"]:
        if name in registry:
            target = datasets(args.votes, registry[name])["full"][2]
        elif name == "Stanford-Politeness":
            target = polite(args.votes)["stack_transfer"][2]
        else:
            target = mbic(args.votes)["full"][2]
        rows = {key(item): (votes[0], *ref) for item, _, _, votes, ref, _ in target}
        with np.load(args.results / name / "target.npz") as data:
            claim, first, second = np.array([rows[int(i)] for i in data["ids"]]).T
            assert np.array_equal(claim != first, data["errors"])
            payload[name] = claim != second
            print(name, len(claim), int((first != second).sum()), flush=True)
    np.savez_compressed(args.out, **payload)


if __name__ == "__main__":
    main()
