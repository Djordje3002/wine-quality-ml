# Data card

UCI Wine Quality, red-wine subset: 1,599 raw rows, 11 numerical predictors and one 0–10 quality score. This run retains 1,359 unique full rows. Records contain laboratory measurements rather than prices, brands, or vintages.

## Source and citation

[Original dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Wine Quality. UCI Machine Learning Repository. https://doi.org/10.24432/C56S3T

## License

UCI lists the dataset under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). Credit the creators and source when reusing data or derived figures. The repository's MIT license applies to its code, not to the dataset.

## Integrity and storage

```text
{
  "url": "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
  "sha256": "3ed56667f4b828242bd732d7d1dd7f2861e54432239d7fa63877014cbb0304d4"
}
```

For downloaded datasets, every run verifies archive bytes against this SHA-256 before parsing. The downloader has a 60-second timeout and 20 MB response cap. It reads named members without extracting filesystem paths. Raw files stay in the ignored `data/raw/` cache. To recover from a corrupted cache, remove `data/raw/source.zip`, verify the upstream source, then rerun training. Do not silently replace the expected hash after a source change.

## Processing and sampling

Exact duplicate rows are removed first. Identical feature vectors share a hash group, including cases with different target scores. A seeded group split reserves the test set; five-fold grouped cross-validation on the training partition chooses the lowest mean MAE. Scaling is fitted inside each pipeline and fold. The selected model is fitted once on training data, then evaluated on the reserved test partition.

## Representation and privacy

One regional wine population and a small dataset limit external validity. No vineyard, batch, or vintage identifiers are available to validate unseen producers or future vintages. Grouping identical measurements removes a specific leakage route; it does not establish independence between all samples. Permutation importance describes this fitted model and holdout, not causal effects.

No private account data, API credentials, or scraped personal profiles are used. Synthetic fixtures exist only in tests and manually constructed examples; published metrics come from the cited dataset.
