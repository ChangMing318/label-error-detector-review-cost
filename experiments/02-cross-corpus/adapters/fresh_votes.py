"""Independent-data replication of fixed-budget item-linked review."""
import argparse
import csv
import gzip
import hashlib
import json
import tarfile
from collections import defaultdict
from io import TextIOWrapper
from pathlib import Path

import numpy as np

from blind_rate import make_rows, score_plans, text_group, write, interval
from external_crowds import feature_models, role_of, table, take
from located_votes import evaluate, queues

P = json.loads(Path(__file__).with_name("fresh_votes_protocol.json").read_text())
ROLES = ("fit", "history", "target")


def unique(rows):
    seen = set()
    for r in rows:
        if r[1] not in seen:
            seen.add(r[1])
            yield r


def polite_file(path):
    items, votes = {}, defaultdict(dict)
    for r in table(path):
        item = f"{r['Community']}:{r['Id']}"
        items[item] = (r["Request"], r["Community"])
        for j in range(1, 6):
            s, w = r[f"Score{j}"], r[f"TurkId{j}"]
            if w and s.isdigit() and 1 <= int(s) <= 25:
                votes[item][w] = int(int(s) > 13)
    return make_rows(items, votes)


def polite(root):
    stack = polite_file(root/"polite_stack-exchange.csv")
    wiki = polite_file(root/"polite_wikipedia.csv")
    all_stack = list(unique(r for k in ROLES for r in stack[k]))
    seen = {r[1] for r in all_stack}
    f = list(unique(r for r in wiki["fit"] if r[1] not in seen))
    h = list(unique(r for role in ("history", "target") for r in wiki[role] if r[1] not in seen))
    return dict(stack_transfer=(f, h, all_stack),
                stack_internal=tuple([r for r in all_stack if role_of(r[1]) == role] for role in ROLES))


def sbic(root):
    items, votes = {}, defaultdict(dict)
    with tarfile.open(root/"SBIC.v2.tgz", "r:gz") as archive:
        for ending, role in (("tst", "target"), ("dev", "history"), ("trn", "fit")):
            suffix = f"SBIC.v2.{ending}.csv"
            m, = (m for m in archive.getmembers() if m.name.split("/")[-1] == suffix)
            with TextIOWrapper(archive.extractfile(m), encoding="utf-8-sig", newline="") as stream:
                for r in csv.DictReader(stream):
                    raw = r["offensiveYN"]
                    if raw not in ("0", "0.0", "0.5", "1", "1.0") or not r["WorkerId"]:
                        continue
                    group = text_group(r["post"])
                    items.setdefault(group, (r["post"], role))
                    if items[group][1] != role:
                        continue
                    votes[group][r["WorkerId"]] = int(float(raw) == 1)
    pooled = make_rows(items, votes)
    official = defaultdict(list)
    for part in pooled.values():
        for row in part:
            official[row[5]].append(row)
    use_official = min(len(official[k]) for k in ROLES) >= 300
    roles = official if use_official else pooled
    return {"official_or_pooled": tuple(roles[k] for k in ROLES)}, (
        "official" if use_official else "pooled")


def mbic(root):
    items, votes = {}, defaultdict(dict)
    with gzip.open(root/"mbic_joined_votes.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        for r in csv.DictReader(stream):
            items[r["id"]] = (r["text"], r["article"])
            votes[r["id"]][r["worker"]] = int(r["label"])
    rows = make_rows(items, votes)
    roles = defaultdict(list)
    for r in unique(row for k in ROLES for row in rows[k]):
        group = text_group(r[5]) if r[5] else r[1]
        roles[role_of(group)].append((r[0], group, *r[2:]))
    return {"full": tuple(roles[k] for k in ROLES)}


def judge_arms(q, shuffled, e, group):
    stats, _ = evaluate(q, shuffled, e, group)
    actual = np.zeros((len(q), len(e)), dtype=np.int8)
    for i, ids in enumerate(q.values()):
        actual[i, ids] = e[ids]
    row = dict(zip(q, actual))
    stats["tests"] = {}
    for name in ("full", "count_then_score"):
        stats["tests"][name] = {c: interval(row[name]-row[c], group) for c in ("votes", "score")}
        stats["tests"][name]["swaps_vs_score"] = dict(added=int((row[name]>row["score"]).sum()),
                                                      dropped=int((row[name]<row["score"]).sum()),
                                                      exchanged=int(np.setdiff1d(q[name], q["score"]).size))
    return stats


def run(args):
    mode = None
    if args.dataset == "sbic":
        cohorts, mode = sbic(args.data)
    else:
        cohorts = (polite if args.dataset == "polite" else mbic)(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    output = {}
    for name, roles in cohorts.items():
        sizes = list(map(len, roles))
        if min(sizes) < 300:
            output[name] = dict(status="below_fixed_support", eligible=sizes, mode=mode)
            continue
        f, h, t = [take(r, k) for r, k in zip(roles, ROLES)]
        public = [r[:4] for r in t]
        names, scores, classes = feature_models(f, h+public)
        frozen, order, _ = score_plans(h, public, scores[:, :len(h)], scores[:, len(h):], classes, names)
        q, shuffled, claim = queues(h, public, scores, classes, frozen, order[:, :frozen["b"]])
        assert all(len(np.unique(ids)) == len(ids) == frozen["b"] for ids in (*q.values(), *shuffled))
        file = args.out/f"{name}.plan.npz"
        np.savez_compressed(file, arms=list(q), orders=np.stack(list(q.values())), shuffled=shuffled)
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        references = np.array([r[4] for r in t])
        e = (references != claim[:, None]).T
        _, group = np.unique([r[1] for r in t], return_inverse=True)
        np.savez_compressed(args.out/f"{name}.judge.npz", errors=e, groups=group)
        result = [judge_arms(q, shuffled, panel, group) for panel in e]
        output[name] = dict(status="complete", eligible=sizes, mode=mode, n=len(t), b=frozen["b"],
                            names=names, choice=frozen["choices"], plan_sha256=digest, refs=result)
        print(args.dataset, name, [r["hits"] for r in result], flush=True)
    write(args.out/"summary.json", output)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=("polite", "sbic", "mbic"))
    for arg in ("data", "out"):
        p.add_argument("--"+arg, type=Path, required=True)
    run(p.parse_args())
