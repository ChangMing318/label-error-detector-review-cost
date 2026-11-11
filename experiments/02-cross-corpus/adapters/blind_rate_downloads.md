# Public input files

Run from the archive root. Download only the files used by the experiments.

```sh
mkdir -p data/raw
for i in 1 2 3; do
  curl -fL --retry 2 "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/full_dataset/goemotions_${i}.csv" -o "data/raw/goemotions_${i}.csv"
done
curl -fL --retry 2 https://ndownloader.figshare.com/files/7554637 -o data/raw/attack_annotated_comments.tsv
curl -fL --retry 2 https://ndownloader.figshare.com/files/7554634 -o data/raw/attack_annotations.tsv
curl -fL --retry 2 https://raw.githubusercontent.com/ishida-lab/irreducible/main/data/fmh_raw.csv -o data/raw/fmh_raw.csv
for f in t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
  curl -fL --retry 2 "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/${f}" -o "data/raw/${f}"
done
```
