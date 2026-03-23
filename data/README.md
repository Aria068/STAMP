# Data

Place dataset CSV files here before running training or evaluation.

## Expected files

| File | Description |
|------|-------------|
| `clean_data/yelp_train_split.csv` | Yelp reviews — training split |
| `clean_data/yelp_test_split.csv` | Yelp reviews — test split |
| `clean_data/filtered_twitter_data_5000.csv` | Twitter posts (5k sample) |
| `clean_data/twitter_test.csv` | Twitter — test split |
| `clean_data/imdb_train_split.csv` | IMDb reviews — training split |
| `clean_data/imdb_test_split.csv` | IMDb reviews — test split |
| `clean_data/SynthPAI_AuthorInfo_train.csv` | SynthPAI attribute inference — train |
| `clean_data/SynthPAI_AuthorInfo_test.csv` | SynthPAI attribute inference — test |

## Required columns

All CSV files must contain at minimum:
- `review` — the raw text
- `author_id` — integer author label (for author-ID evaluation)

SynthPAI additionally requires: `gender`, `age_group`, `education` columns.

## Data sources

- **Yelp**: [Yelp Open Dataset](https://www.yelp.com/dataset)
- **Twitter / IMDb**: standard public NLP benchmark splits
- **SynthPAI**: [yukhymenko2024synthetic](https://huggingface.co/datasets/avykhymenko/SynthPAI)

Create the `clean_data/` directory and place your CSV files there before running any scripts:

```bash
mkdir -p clean_data
```
