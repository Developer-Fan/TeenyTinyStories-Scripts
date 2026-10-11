import pdflatex

compiler = pdflatex.PDFLaTeX.from_texfile("./latex.tex")

compiler.create_pdf(keep_pdf_file=True, keep_log_file=False)