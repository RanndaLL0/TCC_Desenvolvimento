#!/usr/bin/env bash
# Compila o tcc2.tex (com referências) e gera o tcc2.pdf
set -e
cd "$(dirname "$0")"

pdflatex -interaction=nonstopmode tcc2.tex
bibtex tcc2
pdflatex -interaction=nonstopmode tcc2.tex
pdflatex -interaction=nonstopmode tcc2.tex

# Remove os arquivos auxiliares, deixando só o PDF
rm -f tcc2.{aux,bbl,blg,brf,idx,lof,log,lot,out,toc}
