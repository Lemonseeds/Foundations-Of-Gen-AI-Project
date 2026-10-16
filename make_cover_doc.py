"""
Generates the submission Word document and PDF for Step 3 & 4 of the assignment:
YourName_RegNum_FGAI_Assignment2.docx / .pdf

Course: Foundations of Generative AI (ICT4403)
"""
import argparse
import os
import sys
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT


def build_submission_doc(
    student_name: str = "Ayrton Joseph Pulikottil",
    reg_number: str = "230953448",
    github_repo_url: str = "https://github.com/Lemonseeds/Foundations-Of-Gen-AI-Project",
    output_filename: str | None = None,
):
    doc = Document()

    # Page Margins (1 inch all around)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Document Header
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_course = p_title.add_run("FOUNDATIONS OF GENERATIVE AI (ICT4403)\n")
    run_course.font.size = Pt(13)
    run_course.font.bold = True
    run_course.font.color.rgb = RGBColor(38, 70, 83)

    run_sub = p_title.add_run("Assignment 2: Autoregressive Text Generation\nSubmission Cover Sheet\n")
    run_sub.font.size = Pt(17)
    run_sub.font.bold = True
    run_sub.font.color.rgb = RGBColor(42, 157, 143)

    p_div = doc.add_paragraph()
    p_div.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_div.paragraph_format.space_after = Pt(16)
    run_line = p_div.add_run("―" * 48)
    run_line.font.color.rgb = RGBColor(180, 180, 180)

    # Submission Details Table
    details = [
        ("Student Name", student_name),
        ("Registration Number", reg_number),
        ("Assignment Title", "Assignment 2: Comparative Autoregressive Text Generation using Recurrent Neural Networks"),
        (
            "Title of Developed System",
            "Autoregressive Next-Word Prediction & Multi-Temperature Generation System using Simple RNN, Bidirectional RNN, LSTM, and GRU Models",
        ),
        (
            "Name of Dataset Used",
            "The Devil's Dictionary (Ambrose Bierce, 1911) — Project Gutenberg eBook #972",
        ),
        (
            "Dataset Source Link",
            "https://www.gutenberg.org/ebooks/972\n(Plain Text: https://www.gutenberg.org/cache/epub/972/pg972.txt)",
        ),
        ("GitHub Repository Link", github_repo_url),
    ]

    table = doc.add_table(rows=len(details), cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    for i, (label, val) in enumerate(details):
        row = table.rows[i]
        
        # Label cell
        cell_lbl = row.cells[0]
        cell_lbl.width = Inches(2.2)
        p_lbl = cell_lbl.paragraphs[0]
        p_lbl.paragraph_format.space_before = Pt(5)
        p_lbl.paragraph_format.space_after = Pt(5)
        run_l = p_lbl.add_run(label)
        run_l.font.bold = True
        run_l.font.size = Pt(10.5)
        run_l.font.color.rgb = RGBColor(38, 70, 83)

        # Value cell
        cell_val = row.cells[1]
        cell_val.width = Inches(4.3)
        p_val = cell_val.paragraphs[0]
        p_val.paragraph_format.space_before = Pt(5)
        p_val.paragraph_format.space_after = Pt(5)
        run_v = p_val.add_run(val)
        run_v.font.size = Pt(10.5)
        if "http" in val:
            run_v.font.color.rgb = RGBColor(33, 102, 172)
            run_v.font.underline = True

    # Executive Summary of Developed System & Experimental Findings
    p_sec = doc.add_paragraph()
    p_sec.paragraph_format.space_before = Pt(20)
    p_sec.paragraph_format.space_after = Pt(4)
    r_sec = p_sec.add_run("Executive Summary of Developed System & Empirical Findings")
    r_sec.font.bold = True
    r_sec.font.size = Pt(12)
    r_sec.font.color.rgb = RGBColor(38, 70, 83)

    summary_text = (
        "This project implements an end-to-end autoregressive language modeling system comparing four recurrent neural "
        "network architectures: Simple RNN (Elman), Bidirectional RNN (causally restricted to context window representations), "
        "LSTM, and GRU. The models were evaluated on the complete cleaned corpus of Ambrose Bierce's 'The Devil's Dictionary' "
        "(Project Gutenberg eBook #972), an atypical literary dataset featuring dual registers of cynical prose definitions "
        "and mock-heroic verse.\n\n"
        "Experimental Design & Parity:\n"
        "All four architectures were trained under identical conditions: a shared vocabulary of 4,731 tokens, context sequence "
        "length of 20, embedding dimension 128, hidden dimension 256, 2 recurrent layers with dropout 0.3, Adam optimizer (lr=0.003), "
        "batch size 256, gradient norm clipping 1.0, and early stopping on validation cross-entropy loss.\n\n"
        "Key Results & Model Comparison:\n"
        "• Best Model (LSTM): Achieved the lowest test loss (5.7027) and lowest test perplexity (299.67). Its additive memory cell "
        "and gating mechanism prevented gradient decay, allowing effective modeling of long-range syntactic structure.\n"
        "• Runner-Up (GRU): Achieved the highest top-1 prediction accuracy (18.11%) and top-5 accuracy (35.38%) with test perplexity "
        "357.81, training ~27% faster than LSTM (282.1 s vs. 385.3 s).\n"
        "• BiRNN: Attained test perplexity of 373.67, significantly outperforming Simple RNN (424.66) by leveraging forward and backward "
        "sweeps over the context window.\n"
        "• Simple RNN: Had the highest test perplexity (424.66), reflecting the limitations of un-gated recurrence over long contexts.\n\n"
        "Temperature Sampling Analysis:\n"
        "Text generation across seeds ('politics, n.', 'love, n.', 'a lawyer is') at T in {0.2, 0.5, 1.0, 1.5} revealed that T=0.2 "
        "triggers cyclic attractor loops (repetition > 50%), T=0.5 yields optimal grammatical coherence (bigram attestation > 80%), "
        "T=1.0 generates creative diversity (distinct-2 > 0.91) while preserving Bierce's satirical lexicon, and T=1.5 degenerates into semantic noise.\n\n"
        "Deliverables in GitHub Repository:\n"
        "• Master runnable notebook: FGAI_Assignment2.ipynb (fully executed with all outputs, plots, and sample tables inline).\n"
        "• Modular Python source code in src/ (data preprocessing, model architectures, training loop, evaluation metrics).\n"
        "• Trained model checkpoints in models/ (SimpleRNN.pt, BiRNN.pt, LSTM.pt, GRU.pt).\n"
        "• Generated visualizations and numerical metrics in results/.\n"
        "• Detailed, formatted README.md documenting the problem statement, methodology, results, and conclusions."
    )

    p_sum = doc.add_paragraph(summary_text)
    p_sum.paragraph_format.space_after = Pt(10)
    p_sum.paragraph_format.line_spacing = 1.15

    # Determine filename
    if not output_filename:
        safe_name = student_name.replace(" ", "")
        safe_reg = reg_number.replace(" ", "")
        output_filename = f"{safe_name}_{safe_reg}_FGAI_Assignment2.docx"

    doc.save(output_filename)
    print(f"Generated Word document: {output_filename}")
    return output_filename


def convert_docx_to_pdf_via_word(docx_path: str, pdf_path: str | None = None):
    if not pdf_path:
        pdf_path = os.path.splitext(docx_path)[0] + ".pdf"
    
    abs_docx = os.path.abspath(docx_path)
    abs_pdf = os.path.abspath(pdf_path)

    import subprocess
    ps_cmd = f"""
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    try {{
        $doc = $word.Documents.Open('{abs_docx}')
        $doc.SaveAs([ref]'{abs_pdf}', [ref]17)
        $doc.Close()
        Write-Output "PDF saved to {abs_pdf}"
    }} finally {{
        $word.Quit()
    }}
    """
    res = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True, text=True)
    if res.returncode == 0:
        print(f"Successfully converted to PDF: {abs_pdf}")
        return abs_pdf
    else:
        print(f"Error during PDF conversion: {res.stderr}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="Ayrton Joseph Pulikottil", help="Student Name")
    parser.add_argument("--reg", default="230953448", help="Registration Number")
    parser.add_argument("--repo", default="https://github.com/Lemonseeds/Foundations-Of-Gen-AI-Project", help="GitHub repo link")
    parser.add_argument("--out", default=None, help="Output docx filename")
    args = parser.parse_args()

    docx_file = build_submission_doc(args.name, args.reg, args.repo, args.out)
    convert_docx_to_pdf_via_word(docx_file)
