"""PDF report generation from benchmark results."""

import re
from fpdf import FPDF


def clean(text):
    text = re.sub(r'[\x00-\x09\x0b-\x1f]', '', text)
    text = re.sub(r'\[ Prompt:.*?]', '', text)
    for k, v in {'\u2014':'--','\u2013':'-','\u2018':"'",'\u2019':"'",'\u201c':'"','\u201d':'"','\u2026':'...','\u2022':'*','\u2705':'[OK]','\u2192':'->'}.items():
        text = text.replace(k, v)
    text = text.encode('latin-1', errors='replace').decode('latin-1')
    lines = text.strip().split("\n")
    clean_lines = []
    skip = True
    for line in lines:
        if skip and (line.strip().startswith(">") or line.strip() == ""):
            continue
        skip = False
        clean_lines.append(line)
    return "\n".join(clean_lines).strip()


class ReportPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 6, "Bonsai 1-Bit QA Benchmark Report", align="R")
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def generate_pdf(qa_results, mem_results, output_path, timestamp=""):
    """Generate a full benchmark PDF report.

    Args:
        qa_results: dict {model_name: [result_dicts]}
        mem_results: dict {model_name: [mem_dicts]} or None
        output_path: path to write PDF
        timestamp: string timestamp for report
    """
    models = list(qa_results.keys())
    n_questions = len(qa_results[models[0]])

    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Cover ──────────────────────────────────────────────────
    pdf.add_page()
    pdf.ln(25)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(25, 50, 110)
    pdf.cell(0, 14, "Bonsai 1-Bit Model", align="C"); pdf.ln(13)
    pdf.cell(0, 14, "QA Benchmark Report", align="C"); pdf.ln(18)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 7, f"Models: {' | '.join(models)}", align="C"); pdf.ln(7)
    pdf.cell(0, 7, f"{n_questions} Questions | RAG | Finance | Reasoning | Instruction Following", align="C"); pdf.ln(7)
    if timestamp:
        pdf.cell(0, 7, f"Date: {timestamp}", align="C"); pdf.ln(7)

    pdf.ln(10)
    pdf.set_draw_color(25, 50, 110)
    pdf.line(50, pdf.get_y(), 160, pdf.get_y())
    pdf.ln(12)

    # Scorecard
    cats = {}
    for r in qa_results[models[0]]:
        cats.setdefault(r["category"], []).append(r["id"])

    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(25, 50, 110)
    pdf.cell(0, 10, "Score Summary", align="C"); pdf.ln(10)

    cols = ["Model"] + [f"{c} ({len(ids)})" for c, ids in cats.items()] + [f"TOTAL ({n_questions})", "Pass %"]
    widths = [26] + [24] * len(cats) + [24, 18]
    x_start = (210 - sum(widths)) / 2
    pdf.set_x(x_start)
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_fill_color(25, 50, 110)
    pdf.set_text_color(255, 255, 255)
    for c, w in zip(cols, widths):
        pdf.cell(w, 7, c, border=1, align="C", fill=True)
    pdf.ln()

    for mi, m in enumerate(models):
        pdf.set_x(x_start)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(0, 0, 0)
        fill = mi % 2 == 0
        if fill:
            pdf.set_fill_color(232, 238, 248)
        row = [m]
        total = 0
        for cat, ids in cats.items():
            p = sum(1 for r in qa_results[m] if r["category"] == cat and r["accepted"])
            total += p
            row.append(f"{p}/{len(ids)}")
        row.append(f"{total}/{n_questions}")
        row.append(f"{total / n_questions * 100:.0f}%")
        for c, w in zip(row, widths):
            pdf.cell(w, 6, c, border=1, align="C", fill=fill)
        pdf.ln()

    # ── Results grid ───────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(25, 50, 110)
    pdf.cell(0, 10, "1. Per-Question Results"); pdf.ln(8)
    pdf.set_draw_color(25, 50, 110)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(4)

    q_cols = ["Q#", "Category", "Diff", "Description"] + models
    q_w = [8, 22, 12, 46] + [28] * len(models)
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_fill_color(25, 50, 110)
    pdf.set_text_color(255, 255, 255)
    for c, w in zip(q_cols, q_w):
        pdf.cell(w, 6, c, border=1, align="C", fill=True)
    pdf.ln()

    for i in range(n_questions):
        q = qa_results[models[0]][i]
        pdf.set_font("Helvetica", "", 6.5)
        fill = i % 2 == 0
        if fill:
            pdf.set_fill_color(232, 238, 248)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(q_w[0], 5.5, f"Q{q['id']}", border=1, align="C", fill=fill)
        pdf.cell(q_w[1], 5.5, q["category"], border=1, align="C", fill=fill)
        pdf.cell(q_w[2], 5.5, q["difficulty"], border=1, align="C", fill=fill)
        pdf.cell(q_w[3], 5.5, q["description"][:28], border=1, align="L", fill=fill)
        for m in models:
            r = qa_results[m][i]
            if r["accepted"]:
                pdf.set_text_color(0, 120, 0)
                pdf.set_font("Helvetica", "B", 7)
                pdf.cell(q_w[4], 5.5, "PASS", border=1, align="C")
            else:
                pdf.set_text_color(190, 0, 0)
                pdf.set_font("Helvetica", "B", 7)
                pdf.cell(q_w[4], 5.5, "FAIL", border=1, align="C")
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 6.5)
        pdf.ln()

    # ── Memory table ───────────────────────────────────────────
    if mem_results:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(25, 50, 110)
        pdf.cell(0, 10, "2. Memory Usage by Context Size"); pdf.ln(8)
        pdf.set_draw_color(25, 50, 110)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(4)

        m_cols = ["Context"]
        m_w = [20]
        for m in models:
            m_cols += [f"{m} Used", f"{m} KV$"]
            m_w += [28, 28]

        pdf.set_font("Helvetica", "B", 7)
        pdf.set_fill_color(25, 50, 110)
        pdf.set_text_color(255, 255, 255)
        for c, w in zip(m_cols, m_w):
            pdf.cell(w, 6, c, border=1, align="C", fill=True)
        pdf.ln()
        pdf.set_text_color(0, 0, 0)

        labels = [r["context_label"] for r in mem_results[models[0]]]
        for i, label in enumerate(labels):
            row = [label]
            for m in models:
                r = mem_results[m][i]
                row += [f"{r['self_mib']} MiB", f"{r['context_mib']} MiB"]
            pdf.set_font("Helvetica", "", 7)
            fill = i % 2 == 0
            if fill:
                pdf.set_fill_color(232, 238, 248)
            for c, w in zip(row, m_w):
                pdf.cell(w, 5.5, c, border=1, align="C", fill=fill)
            pdf.ln()

    # ── Detailed Q&A ───────────────────────────────────────────
    colors = [(0, 90, 150), (0, 120, 50), (150, 50, 0), (100, 0, 130)]

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(25, 50, 110)
    section_num = "3" if mem_results else "2"
    pdf.cell(0, 10, f"{section_num}. Detailed Q&A Comparison"); pdf.ln(8)
    pdf.set_draw_color(25, 50, 110)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(4)

    for i in range(n_questions):
        q = qa_results[models[0]][i]

        if pdf.get_y() > 210:
            pdf.add_page()

        # Header
        pdf.set_fill_color(232, 238, 248)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(25, 50, 110)
        pdf.cell(0, 6, f"Q{q['id']}  [{q['category']}] [{q['difficulty']}] {q['description']}", fill=True)
        pdf.ln(7)

        # Question (truncated)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(40, 40, 40)
        qtext = clean(q["question"])
        if len(qtext) > 400:
            qtext = qtext[:400] + "..."
        pdf.set_x(12)
        pdf.multi_cell(186, 3.8, qtext)
        pdf.ln(2)

        # Answers
        for mi, m in enumerate(models):
            r = qa_results[m][i]
            cr, cg, cb = colors[mi % len(colors)]

            pdf.set_font("Helvetica", "B", 7.5)
            pdf.set_text_color(cr, cg, cb)
            pdf.cell(25, 4.5, f"  {m}")
            if r["accepted"]:
                pdf.set_text_color(0, 120, 0)
                pdf.cell(10, 4.5, "[PASS]")
            else:
                pdf.set_text_color(180, 0, 0)
                pdf.cell(10, 4.5, "[FAIL]")
            pdf.set_font("Helvetica", "", 6.5)
            pdf.set_text_color(120, 120, 120)
            pdf.cell(0, 4.5, f"Wall={r['wall_time_s']:.1f}s  Gen={r['gen_tps']:.0f}t/s")
            pdf.ln(4.5)

            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(50, 50, 50)
            ans = clean(r["answer"])
            if len(ans) > 400:
                ans = ans[:400] + "..."
            pdf.set_x(14)
            pdf.multi_cell(182, 3.5, ans)
            pdf.ln(1.5)

        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)

    pdf.output(output_path)
    return output_path
