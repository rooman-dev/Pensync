import sys
from pathlib import Path

try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.units import mm
except Exception as e:
    print("Missing dependency: reportlab is required. Install with: pip install reportlab")
    raise


def md_to_pdf(md_path, pdf_path):
    text = Path(md_path).read_text(encoding="utf-8")
    lines = text.splitlines()

    doc = SimpleDocTemplate(str(pdf_path), pagesize=landscape(A4), rightMargin=20*mm, leftMargin=20*mm, topMargin=15*mm, bottomMargin=15*mm)
    styles = getSampleStyleSheet()
    normal = styles['Normal']
    h1 = styles['Heading1']
    h2 = styles['Heading2']

    flow = []

    for line in lines:
        if line.startswith('# '):
            flow.append(Paragraph(line[2:].strip(), h1))
        elif line.startswith('## '):
            flow.append(Paragraph(line[3:].strip(), h2))
        elif line.startswith('- '):
            flow.append(Paragraph('• ' + line[2:].strip(), normal))
        elif line.strip() == '':
            flow.append(Spacer(1, 6))
        else:
            flow.append(Paragraph(line.strip(), normal))

    doc.build(flow)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python tools/generate_pdf_report.py <input.md> <output.pdf>")
        sys.exit(1)
    md_to_pdf(sys.argv[1], sys.argv[2])
    print(f"Wrote PDF: {sys.argv[2]}")
