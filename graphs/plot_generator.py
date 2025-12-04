"""Utility script to generate simple trend or histogram charts from user input.

The script prompts for:
- The desired chart type (trend line or histogram)
- A title describing the metric being plotted
- Variant names for the x-axis
- Floating-point values for the y-axis
- An optional output path for the generated image
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def _prompt_plot_type() -> str:
    """Prompt the user for the desired plot type."""
    options = {"t": "trend", "trend": "trend", "h": "histogram", "hist": "histogram", "histogram": "histogram"}
    while True:
        choice = input(
            "Scegli il tipo di grafico: 'trend' per una curva di tendenza o 'histogram' per un istogramma (t/h): "
        ).strip().lower()
        if choice in options:
            return options[choice]
        print("Scelta non valida. Inserisci 'trend' o 'histogram'.")


def _prompt_title() -> str:
    """Prompt the user for the chart title."""
    while True:
        title = input("Inserisci il titolo del grafico: ").strip()
        if title:
            return title
        print("Il titolo non può essere vuoto.")

def _prompt_ylabel() -> str:
    """Prompt the user for the chart ylabel."""
    while True:
        ylabel = input("Inserisci il titolo delle ordinate (metrica di interesse): ").strip()
        if ylabel:
            return ylabel
        print("La descrizione delle ordinate non può essere vuota.")


def _prompt_variants() -> List[str]:
    """Prompt for variant names used on the x-axis."""
    while True:
        raw = input("Inserisci i nomi delle varianti separati da virgola: ").strip()
        names = [name.strip() for name in raw.split(",") if name.strip()]
        if names:
            return names
        print("Devi fornire almeno una variante.")


def _prompt_values(count: int) -> List[float]:
    """Prompt for floating point values matching the variant count."""
    while True:
        raw = input(
            "Inserisci i valori float separati da virgola nello stesso ordine delle varianti: "
        ).strip()
        try:
            values = [float(value.strip()) for value in raw.split(",") if value.strip()]
        except ValueError:
            print("Tutti i valori devono essere numeri decimali. Riprova.")
            continue

        if len(values) != count:
            print(f"Numero di valori non valido. Attesi {count}, trovati {len(values)}. Riprova.")
            continue
        return values


def _prompt_output_path(default_name: str = "grafico.png") -> Path:
    """Prompt for an optional output path; fall back to the provided default."""
    raw_path = input(
        f"Percorso file di output (ad esempio output.png). Lascia vuoto per usare '{default_name}': "
    ).strip()
    return Path(raw_path) if raw_path else Path(default_name)


def _plot_trend(ylabel: str, variants: List[str], values: List[float], title: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(variants)), values, marker="o", linestyle="-", color="#1f77b4")
    plt.xticks(ticks=range(len(variants)), labels=variants, rotation=45, ha="right")
    plt.xlabel("Epochs") #change into Variants when needed
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Grafico di tendenza salvato in: {output_path}")


def _plot_histogram(ylabel: str, variants: List[str], values: List[float], title: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.bar(variants, values, color="#ff7f0e")
    plt.xlabel("Epochs") #change into Variants when needed
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Istogramma salvato in: {output_path}")


def main() -> None:
    print("Generatore di grafici (trend o istogramma)")
    plot_type = _prompt_plot_type()
    title = _prompt_title()
    ylabel = _prompt_ylabel()
    variants = _prompt_variants()
    values = _prompt_values(len(variants))
    output_path = _prompt_output_path()

    if plot_type == "trend":
        _plot_trend(ylabel, variants, values, title, output_path)
    else:
        _plot_histogram(ylabel, variants, values, title, output_path)


if __name__ == "__main__":
    main()