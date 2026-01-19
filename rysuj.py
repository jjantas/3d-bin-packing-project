import pandas as pd
import glob
import matplotlib.pyplot as plt
import os
import re
import argparse

# ================= KONFIGURACJA =================
SCIEZKA_PLIKOW = "runs/convergence/*.csv"
# ================================================

def wyciagnij_parametry(nazwa):
    """Parsuje nazwę pliku/run_id, aby wyciągnąć parametry do legendy."""
    rtr = re.search(r'rtr([\d\.]+)', nazwa)
    pm = re.search(r'pm([\d\.]+)', nazwa)
    init = re.search(r'init([\d\.]+)', nazwa)
    pc = re.search(r'pc([\d\.]+)', nazwa)
    prs = re.search(r'prs([\d\.]+)', nazwa)

    return {
        "rtr": rtr.group(1) if rtr else "-",
        "pm": pm.group(1) if pm else "-",
        "init": init.group(1) if init else "-",
        "pc": pc.group(1) if pc else "-",
        "prs": prs.group(1) if prs else "-"
    }

def generuj_wykres(df_calosc, wybrane_gridy, kolumna_y, tytul_wykresu, nazwa_pliku):
    """
    Uniwersalna funkcja rysująca.
    Przyjmuje:
    - df_calosc: pełne dane
    - wybrane_gridy: lista run_id, które chcemy narysować (np. top 20 lub bottom 20)
    - kolumna_y: co rysujemy (avg_report/best_report)
    - tytul_wykresu: nagłówek wykresu
    - nazwa_pliku: gdzie zapisać plik png
    """
    
    # Filtrujemy dane tylko dla wybranych gridów
    df_subset = df_calosc[df_calosc['Etykieta'].isin(wybrane_gridy)].copy()
    
    # Agregacja: średnia po seedach dla każdej generacji
    # (Dzięki temu mamy jedną linię na konfigurację, a nie 3 poszarpane)
    df_agg = df_subset.groupby(['Etykieta', 'gen'])[kolumna_y].mean().reset_index()

    # Opcjonalne przycięcie do 80 generacji (z Twojego kodu)
    df_agg = df_agg[df_agg['gen'] <= 80]

    plt.figure(figsize=(14, 8))

    # Iterujemy po wybranych gridach, żeby narysować linie
    # Sortujemy wybrane_gridy, żeby legenda była w miarę uporządkowana
    for grid_id in wybrane_gridy:
        dane_linii = df_agg[df_agg["Etykieta"] == grid_id]
        
        if dane_linii.empty:
            continue

        # Tworzenie czytelnej etykiety
        params = wyciagnij_parametry(grid_id)
        label = (f"rtr:{params['rtr']} pm:{params['pm']} "
                 f"prs:{params['prs']} init:{params['init']} pc:{params['pc']}")

        plt.plot(dane_linii['gen'], dane_linii[kolumna_y], label=label, linewidth=1.5)

    plt.title(tytul_wykresu)
    plt.xlabel('Generacja')
    plt.ylabel(f'{kolumna_y} funkcji celu')
    plt.xlim(0, 80) # Opcjonalne, jeśli chcesz widzieć tylko początek
    
    # Legenda - przy wielu liniach warto ją zmniejszyć lub wyrzucić na zewnątrz
    plt.legend(title='Parametry (rtr, pm, prs, init, pc)', loc='upper left', bbox_to_anchor=(1, 1), fontsize='small', ncol=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout() # Ważne, żeby legenda nie została ucięta
    
    print(f"Zapisywanie wykresu: {nazwa_pliku}")
    plt.savefig(nazwa_pliku, dpi=300)
    # plt.show() # Odkomentuj, jeśli chcesz, żeby się wyświetlało okno (może blokować pętlę)
    plt.close() # Zamykamy figurę, żeby zwolnić pamięć przed następnym wykresem

def main(kolumna_y, n_top):
    pliki = glob.glob(SCIEZKA_PLIKOW)
    if not pliki:
        print("Błąd: Nie znaleziono żadnych plików CSV.")
        return

    print(f"Znaleziono {len(pliki)} plików. Wczytywanie i przetwarzanie...")

    lista_df = []
    for plik in pliki:
        try:
            df = pd.read_csv(plik)
            if 'run_id' in df.columns:
                df['Etykieta'] = df['run_id']
                # Wybieramy minimalny zestaw kolumn dla oszczędności pamięci
                subset = df[['gen', kolumna_y, 'Etykieta']].copy()
                lista_df.append(subset)
        except Exception as e:
            print(f"Błąd odczytu {plik}: {e}")

    if not lista_df:
        print("Brak danych.")
        return

    df_calosc = pd.concat(lista_df, ignore_index=True)

    # --- TWORZENIE RANKINGU ---
    # Grupujemy po konfiguracji i bierzemy maksymalny wynik jaki kiedykolwiek osiągnęła
    # (Zakładamy, że im więcej tym lepiej. Jeśli im mniej tym lepiej -> zmień ascending na True)
    ranking = df_calosc.groupby('Etykieta')[kolumna_y].max().sort_values(ascending=False)
    
    # 1. Wybieramy NAJLEPSZE
    top_grids = ranking.head(n_top).index.tolist()
    
    # 2. Wybieramy NAJGORSZE
    worst_grids = ranking.tail(n_top).index.tolist()

    print(f"Generowanie wykresów dla TOP {n_top} i WORST {n_top}...")

    # Rysujemy Najlepsze
    generuj_wykres(
        df_calosc, 
        top_grids, 
        kolumna_y, 
        f"TOP {n_top} Najlepszych Konfiguracji ({kolumna_y})", 
        f"convergence_TOP_{n_top}_{kolumna_y}.png"
    )

    # Rysujemy Najgorsze
    generuj_wykres(
        df_calosc, 
        worst_grids, 
        kolumna_y, 
        f"BOTTOM {n_top} Najgorszych Konfiguracji ({kolumna_y})", 
        f"convergence_WORST_{n_top}_{kolumna_y}.png"
    )

    print("Gotowe.")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Rysowanie wykresów Best vs Worst')
    argparser.add_argument('--rysuj', choices=['avg_report', 'best_report'], default='avg_report', help='Metryka do rysowania')
    argparser.add_argument('--top', type=int, default=10, help='Ile konfiguracji pokazać na wykresach')
    
    args = argparser.parse_args()
    main(args.rysuj, args.top)