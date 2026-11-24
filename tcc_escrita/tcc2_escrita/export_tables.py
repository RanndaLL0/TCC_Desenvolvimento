#!/usr/bin/env python3
"""Exporta cada tabela do tcc2.tex como PDF, PNG e SVG.

Cada imagem inclui o título ("Tabela N – ...") acima e a fonte abaixo, com a
mesma formatação do TCC (classe abntex2 e configuração de legendas do tcc2.tex).
O número N é a posição da tabela no documento.

O nome dos arquivos segue o \\label da tabela, sem o prefixo "tab:"
(ex.: \\label{tab:lacunas} -> images/tabelas/lacunas.{pdf,png,svg}).

Uso:
    python3 export_tables.py                  # todas as tabelas
    python3 export_tables.py lacunas divisao  # só as indicadas
    python3 export_tables.py --dpi 300 --out /outro/diretorio

Requer: pdflatex, pdfcrop, pdftoppm e pdftocairo (poppler-utils).
"""
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "tcc2.tex"

# Mesma classe e pacotes do tcc2.tex, para que título, fonte e numeração saiam
# idênticos ao PDF do TCC (margens ABNT: texto com 16 cm de largura).
PREAMBLE = r"""\documentclass[12pt,a4paper,brazil]{abntex2}
\usepackage{lmodern}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{tabularx}
\usepackage{graphicx}
%(captions)s
\pagestyle{empty}
\begin{document}
\setcounter{table}{%(number)d}
%(body)s
\end{document}
"""

# Trecho do tcc2.tex que ajusta as legendas para 10 pt (título e "Fonte:").
CAPTION_START = r"\renewcommand{\ABNTEXfontereduzida}"
CAPTION_END = r"\endgroup}" + "\n" + r"\makeatother"

TABLE_ENV = re.compile(r"\\begin\{table\}(?:\[[^\]]*\])?(.*?)\\end\{table\}", re.S)
LABEL = re.compile(r"\\label\{tab:([^}]+)\}")


def caption_setup(source: Path) -> str:
    """Copia do tcc2.tex a configuração de legendas, para acompanhar mudanças nela."""
    tex = source.read_text(encoding="utf-8")
    i = tex.find(CAPTION_START)
    j = tex.find(CAPTION_END, i)
    if i < 0 or j < 0:
        raise SystemExit("erro: configuração de legendas não encontrada no tcc2.tex")
    return tex[i : j + len(CAPTION_END)]


def extract_tables(source: Path) -> dict[str, str]:
    tex = source.read_text(encoding="utf-8")
    # Ignora comentários de linha inteira para não capturar exemplos comentados.
    tex = "\n".join(l for l in tex.splitlines() if not l.lstrip().startswith("%"))
    tables: dict[str, str] = {}
    for m in TABLE_ENV.finditer(tex):
        label = LABEL.search(m.group(1))
        if not label:
            print("aviso: tabela sem \\label{tab:...}, ignorada", file=sys.stderr)
            continue
        name = label.group(1)
        if name in tables:
            raise SystemExit(f"erro: label duplicado tab:{name}")
        tables[name] = m.group(0)  # ambiente completo: título, tabela e fonte
    return tables


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.DEVNULL)


def export(name: str, number: int, body: str, captions: str, out_dir: Path, dpi: int) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / f"{name}.tex").write_text(
            PREAMBLE % {"captions": captions, "number": number - 1, "body": body},
            encoding="utf-8",
        )
        try:
            run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"], work)
        except subprocess.CalledProcessError:
            log = (work / f"{name}.log").read_text(errors="replace")
            raise SystemExit(f"erro ao compilar tab:{name}:\n" + "\n".join(log.splitlines()[-25:]))
        # Recorta a página inteira no conteúdo (título + tabela + fonte).
        run(["pdfcrop", "--margins", "6", f"{name}.pdf", f"{name}-crop.pdf"], work)
        (work / f"{name}-crop.pdf").replace(work / f"{name}.pdf")
        run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile", f"{name}.pdf", name], work)
        run(["pdftocairo", "-svg", f"{name}.pdf", f"{name}.svg"], work)
        for ext in ("pdf", "png", "svg"):
            shutil.copy(work / f"{name}.{ext}", out_dir / f"{name}.{ext}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", nargs="*", help="labels (sem 'tab:') a exportar; vazio = todas")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--out", type=Path, default=HERE / "images" / "tabelas")
    args = ap.parse_args()

    for tool in ("pdflatex", "pdfcrop", "pdftoppm", "pdftocairo"):
        if not shutil.which(tool):
            raise SystemExit(f"erro: '{tool}' não encontrado no PATH")

    tables = extract_tables(SOURCE)
    wanted = args.labels or list(tables)
    unknown = [w for w in wanted if w not in tables]
    if unknown:
        raise SystemExit(f"labels inexistentes: {unknown}. Disponíveis: {list(tables)}")

    args.out.mkdir(parents=True, exist_ok=True)
    captions = caption_setup(SOURCE)
    numbers = {n: i for i, n in enumerate(tables, start=1)}
    for name in wanted:
        export(name, numbers[name], tables[name], captions, args.out, args.dpi)
        print(f"ok: {args.out / name}.{{pdf,png,svg}}")


if __name__ == "__main__":
    main()
