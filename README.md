# adhd-eeg-classifier

Binarny klasyfikator, który na podstawie nagrań EEG odróżnia dzieci z ADHD
od grupy kontrolnej. Dane pochodzą ze zbioru Kaggle
`danizo/eeg-dataset-for-adhd`. Projekt powstał jako zadanie rekrutacyjne do
sekcji AI koła naukowego KN Neuron na Politechnice Wrocławskiej.

Projekt jest w trakcie pracy. Wyniki, decyzje i ograniczenia opiszę tu, kiedy
będą liczby, które je potwierdzą.

## Uruchomienie

```bash
uv sync
uv run python -m scripts.download_data
```

Skrypt pobiera z Kaggle archiwum (47 MB) i rozpakowuje plik `adhdata.csv`
(267 MB) do `data/raw/`.
Jeśli w `~/.kaggle/kaggle.json` jest token API Kaggle, `kagglehub` go użyje.
