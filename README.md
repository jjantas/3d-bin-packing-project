# 3D Bin Packing – Optymalizacja Pakowania Paczek 📦

Ten projekt to program, który rozwiązuje problem pakowania 3D. Jego zadaniem jest zmieszczenie jak największej liczby mniejszych pudełek (paczek) w jednym dużym magazynie (kontenerze), przestrzegając zasad fizyki (paczki nie mogą na siebie wchodzić ani wisieć w powietrzu).

## Opis działania

Program korzysta z dwóch głównych metod:

1.  **Algorytm Genetyczny (GA):** Działa podobnie do ewolucji. Program tworzy wiele losowych ułożeń, wybiera te najlepsze, miesza je ze sobą i wprowadza drobne zmiany (mutacje). Dzięki temu z każdym "pokoleniem" ułożenie paczek jest coraz lepsze.
2.  **Przeszukiwanie Losowe:** Program próbuje układać paczki losowo wiele razy i wybiera najlepszą próbę. Służy to głównie jako punkt odniesienia, żeby sprawdzić, czy Algorytm Genetyczny działa skutecznie.

## Instrukcja

Głównym plikiem, który uruchamiasz do pojedynczego pakowania, jest `main.py`.

### 1. Podstawowe uruchomienie
Uruchamia program z domyślnymi ustawieniami i pokazuje wynik w 3D.

    python main.py --plot

### 2. Własne paczki (plik CSV)
Możesz wczytać listę własnych pudełek z pliku CSV. Plik musi zawierać kolumny `l,w,h` (długość, szerokość, wysokość).

    python main.py --boxes_csv sciezka/do/pliku.csv --plot


## Benchmark i Testowanie Wydajności

Program posiada moduł do **automatycznych testów**. Służy on do sprawdzania, jak dobrze algorytm radzi sobie z różnymi zestawami paczek oraz do szukania idealnych ustawień.

Zamiast uruchamiać program ręcznie, użyj flagi `--benchmark`.

**1. Tryb podstawowy (Porównanie)**
Uruchamia porównanie Algorytmu Genetycznego z losowym układaniem. Wyniki zapisują się w `runs/summary.csv`.

    python main.py --benchmark

**2. Tryb strojenia (`--tuning`)**
Uruchamia tzw. **Grid Search**. Program sprawdzi setki kombinacji parametrów (może to zająć dużo czasu).

    python main.py --benchmark --tuning

## Analiza i Wykresy (Plot Results)

Gdy już przeprowadzisz testy (benchmark), możesz użyć skryptu `plot_results.py`, aby zamienić surowe liczby w czytelne wykresy 2D.

### Jak generować wykresy?

Uruchom poniższą komendę po zakończeniu benchmarku:
```python
python main.py --benchmark --tuning=True --warehouse 35 35 35 --boxes_csv data/boxes_2.csv
python plot_results.py --mode hyperparams --conv_glob "runs/convergence/A/*.csv" --metric best_report
```
### Co się wydarzy?
Program przeanalizuje pliki z folderu `runs/` i utworzy nowy folder `runs/plots/` zawierający:

1.  **Porównanie (Summary):** Wykresy kropkowe pokazujące, o ile lepiej algorytm genetyczny radzi sobie od losowego (`summary_scatter_fitness.png`).
2.  **Ranking:** Wykres słupkowy pokazujący, które ustawienia (konfiguracje) były najlepsze (`ga_configs_ranking.png`).
3.  **Wykresy postępu (Convergence):**  Linie pokazujące, jak szybko algorytm "uczył się" układać paczki w kolejnych pokoleniach.

**Opcje dodatkowe:**
Możesz wybrać konkretny tryb analizy flagą `--mode`:
* `--mode summary` – tylko ogólne porównanie wyników.
* `--mode hyperparams` – analiza szczegółowa (dla trybu `--tuning`).

## Najważniejsze parametry (main.py)

Możesz sterować działaniem programu, dodając te opcje przy uruchamianiu:

| Parametr | Opis w prostym języku |
| :--- | :--- |
| `--plot` | Wyświetla wizualizację 3D pojedynczego rozwiązania. |
| `--pop` | **Wielkość populacji**. Ile różnych ułożeń program sprawdza naraz. |
| `--gen` | **Liczba generacji**. Ile razy program ma ulepszać rozwiązania. |
| `--seed` | **Ziarno losowości**. Stała liczba pozwala uzyskać ten sam wynik (powtarzalność). |
| `--patience` | **Cierpliwość**. Jeśli wynik nie poprawi się przez tyle tur, program kończy pracę. |

## Struktura plików (Co jest czym?)

* **`main.py`** – Główny plik sterujący (uruchamianie i konfiguracja).
* **`plot_results.py`** – Narzędzie do tworzenia wykresów 2D i analizy wyników benchmarków.
* **`ga.py`** – "Mózg" programu (algorytm genetyczny).
* **`benchmark.py`** – Moduł obsługujący benchmarking.
* **`fitness.py`** – "Sędzia" (ocena ułożenia paczek).
* **`experiments.py`** – Logika układania (fizyka pakowania).
* **`viz.py`** – Wizualizacja 3D (dla pojedynczych rozwiązań).
