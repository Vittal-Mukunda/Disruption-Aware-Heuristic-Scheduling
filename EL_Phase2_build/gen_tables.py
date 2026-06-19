# -*- coding: utf-8 -*-
"""Generate the 4 Phase-II table slides with styling identical to the template's
Literature Review table (Bookman Old Style, navy 002060 text, thin dk1 borders),
plus a light header fill, a highlighted winner row, and a merged 'takeaway' row.
Writes full <p:sld> XML for slide6 (Main Results) and slide16/17/18."""
import os

SLIDES_DIR = r"C:\Users\vitta\OneDrive\Desktop\DAHS\EL_Phase2_build\unpacked\ppt\slides"
DATE = "Tuesday, 02 June 2026"
NAVY = "002060"
RED = "C00000"
FONT = "Bookman Old Style"
HEADER_FILL = "D9E1F2"
WIN_FILL = "E2EFDA"
NOTE_FILL = "F2F2F2"

HEADER = ('<?xml version="1.0" encoding="utf-8"?>\n'
'<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
'xmlns:mv="urn:schemas-microsoft-com:mac:vml" '
'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
'xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" '
'xmlns:dgm="http://schemas.openxmlformats.org/drawingml/2006/diagram" '
'xmlns:o="urn:schemas-microsoft-com:office:office" '
'xmlns:v="urn:schemas-microsoft-com:vml" '
'xmlns:pvml="urn:schemas-microsoft-com:office:powerpoint" '
'xmlns:com="http://schemas.openxmlformats.org/drawingml/2006/compatibility" '
'xmlns:p14="http://schemas.microsoft.com/office/powerpoint/2010/main" '
'xmlns:p15="http://schemas.microsoft.com/office/powerpoint/2012/main" '
'xmlns:ahyp="http://schemas.microsoft.com/office/drawing/2018/hyperlinkcolor">')


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def border(tag):
    return ('<a:%s cap="flat" cmpd="sng" w="9525"><a:solidFill><a:schemeClr val="dk1"/>'
            '</a:solidFill><a:prstDash val="solid"/><a:round/>'
            '<a:headEnd len="sm" w="sm" type="none"/>'
            '<a:tailEnd len="sm" w="sm" type="none"/></a:%s>') % (tag, tag)


def para(text, sz, bold, color, align):
    runs = ""
    parts = text.split("\n")
    out = []
    for part in parts:
        rpr = ('<a:rPr b="1" lang="en-GB" sz="%d">' % sz) if bold else ('<a:rPr lang="en-GB" sz="%d">' % sz)
        run = (rpr +
               '<a:solidFill><a:srgbClr val="%s"/></a:solidFill>'
               '<a:latin typeface="%s"/><a:ea typeface="%s"/><a:cs typeface="%s"/><a:sym typeface="%s"/>'
               '</a:rPr><a:t>%s</a:t>' % (color, FONT, FONT, FONT, FONT, esc(part)))
        p = ('<a:p><a:pPr indent="0" lvl="0" marL="0" rtl="0" algn="%s">'
             '<a:lnSpc><a:spcPct val="100000"/></a:lnSpc>'
             '<a:spcBef><a:spcPts val="0"/></a:spcBef><a:spcAft><a:spcPts val="0"/></a:spcAft>'
             '<a:buNone/></a:pPr><a:r>%s</a:r></a:p>' % (align, run))
        out.append(p)
    return "".join(out)


def cell(text, sz=1200, bold=False, color=NAVY, align="l", fill=None, gridspan=None, hmerge=False):
    if hmerge:
        return '<a:tc hMerge="1"><a:txBody><a:bodyPr/><a:lstStyle/><a:p><a:endParaRPr/></a:p></a:txBody><a:tcPr/></a:tc>'
    span_attr = (' gridSpan="%d"' % gridspan) if gridspan else ""
    body = para(text, sz, bold, color, align)
    fillxml = ('<a:solidFill><a:srgbClr val="%s"/></a:solidFill>' % fill) if fill else ""
    tcpr = ('<a:tcPr marT="54000" marB="54000" marR="73000" marL="73000" anchor="ctr">'
            + border("lnL") + border("lnR") + border("lnT") + border("lnB") + fillxml + '</a:tcPr>')
    return ('<a:tc%s><a:txBody><a:bodyPr/><a:lstStyle/>%s</a:txBody>%s</a:tc>'
            % (span_attr, body, tcpr))


def make_slide(title, title_sz, col_widths, rows, caption, idbase=400):
    total = sum(col_widths)
    offx = (9144000 - total) // 2
    offy = 880000
    grid = "".join('<a:gridCol w="%d"/>' % w for w in col_widths)
    trs = ""
    for row in rows:
        h = row.get("h", 360000)
        trs += '<a:tr h="%d">' % h + "".join(row["cells"]) + "</a:tr>"
    table = ('<p:graphicFrame><p:nvGraphicFramePr>'
             '<p:cNvPr id="%d" name="Table"/><p:cNvGraphicFramePr/><p:nvPr/>'
             '</p:nvGraphicFramePr>'
             '<p:xfrm><a:off x="%d" y="%d"/><a:ext cx="%d" cy="3000000"/></p:xfrm>'
             '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table">'
             '<a:tbl><a:tblPr><a:noFill/>'
             '<a:tableStyleId>{C7660BB1-E7F1-4267-BE83-C20300E01D2A}</a:tableStyleId></a:tblPr>'
             '<a:tblGrid>%s</a:tblGrid>%s</a:tbl></a:graphicData></a:graphic></p:graphicFrame>'
             % (idbase + 3, offx, offy, total, grid, trs))

    # title box (white bg, dashed, red Bookman centred) -- mirrors template
    titlebox = ('<p:sp><p:nvSpPr><p:cNvPr id="%d" name="Title"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
                '<p:spPr><a:xfrm><a:off x="611704" y="67310"/><a:ext cx="7920592" cy="707390"/></a:xfrm>'
                '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:solidFill><a:schemeClr val="lt1"/></a:solidFill>'
                '<a:ln cap="flat" cmpd="sng" w="9525"><a:solidFill><a:schemeClr val="lt1"/></a:solidFill>'
                '<a:prstDash val="dash"/><a:miter lim="800000"/>'
                '<a:headEnd len="sm" w="sm" type="none"/><a:tailEnd len="sm" w="sm" type="none"/></a:ln></p:spPr>'
                '<p:txBody><a:bodyPr anchorCtr="0" anchor="ctr" bIns="34275" lIns="68575" spcFirstLastPara="1" '
                'rIns="68575" wrap="square" tIns="34275"><a:noAutofit/></a:bodyPr><a:lstStyle/>'
                '<a:p><a:pPr indent="0" lvl="0" marL="0" marR="0" rtl="0" algn="ctr">'
                '<a:spcBef><a:spcPts val="0"/></a:spcBef><a:spcAft><a:spcPts val="0"/></a:spcAft><a:buNone/></a:pPr>'
                '<a:r><a:rPr b="1" lang="en-GB" sz="%d"><a:solidFill><a:srgbClr val="%s"/></a:solidFill>'
                '<a:latin typeface="%s"/><a:ea typeface="%s"/><a:cs typeface="%s"/><a:sym typeface="%s"/></a:rPr>'
                '<a:t>%s</a:t></a:r></a:p></p:txBody></p:sp>'
                % (idbase + 1, title_sz, RED, FONT, FONT, FONT, FONT, esc(title)))

    # slide number + date footer
    sldnum = ('<p:sp><p:nvSpPr><p:cNvPr id="%d" name="num"/><p:cNvSpPr txBox="1"/>'
              '<p:nvPr><p:ph idx="12" type="sldNum"/></p:nvPr></p:nvSpPr>'
              '<p:spPr><a:xfrm><a:off x="6955491" y="4760404"/><a:ext cx="2057400" cy="273844"/></a:xfrm>'
              '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>'
              '<p:txBody><a:bodyPr anchorCtr="0" anchor="ctr" bIns="34275" lIns="68575" spcFirstLastPara="1" '
              'rIns="68575" wrap="square" tIns="34275"><a:noAutofit/></a:bodyPr><a:lstStyle/>'
              '<a:p><a:pPr indent="0" lvl="0" marL="0" rtl="0" algn="r"><a:spcBef><a:spcPts val="0"/></a:spcBef>'
              '<a:spcAft><a:spcPts val="0"/></a:spcAft><a:buNone/></a:pPr>'
              '<a:fld id="{00000000-1234-1234-1234-123412341234}" type="slidenum"><a:rPr lang="en-GB"/>'
              '<a:t>‹#›</a:t></a:fld><a:endParaRPr/></a:p></p:txBody></p:sp>' % (idbase + 4))

    datebox = ('<p:sp><p:nvSpPr><p:cNvPr id="%d" name="dt"/><p:cNvSpPr txBox="1"/>'
               '<p:nvPr><p:ph idx="10" type="dt"/></p:nvPr></p:nvSpPr>'
               '<p:spPr><a:xfrm><a:off x="131109" y="4760404"/><a:ext cx="2057400" cy="273900"/></a:xfrm>'
               '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>'
               '<p:txBody><a:bodyPr anchorCtr="0" anchor="ctr" bIns="34275" lIns="68575" spcFirstLastPara="1" '
               'rIns="68575" wrap="square" tIns="34275"><a:noAutofit/></a:bodyPr><a:lstStyle/>'
               '<a:p><a:pPr indent="0" lvl="0" marL="0" rtl="0" algn="ctr"><a:spcBef><a:spcPts val="0"/></a:spcBef>'
               '<a:spcAft><a:spcPts val="0"/></a:spcAft><a:buNone/></a:pPr>'
               '<a:r><a:rPr lang="en-GB"/><a:t>%s</a:t></a:r><a:endParaRPr/></a:p></p:txBody></p:sp>' % (idbase + 5, DATE))

    sptree = ('<p:nvGrpSpPr><p:cNvPr id="%d" name="Shape"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
              '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
              '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
              % idbase) + sldnum + titlebox + table + datebox

    return (HEADER + '<p:cSld><p:spTree>' + sptree + '</p:spTree></p:cSld>'
            '<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>')


def hrow(labels, widths, sz=1200):
    return {"h": 380000, "cells": [cell(t, sz=sz, bold=True, color=NAVY, align="ctr", fill=HEADER_FILL) for t in labels]}


def drow(vals, aligns, sz=1200, fill=None, bold=False, h=360000):
    return {"h": h, "cells": [cell(v, sz=sz, bold=bold, color=NAVY, align=a, fill=fill) for v, a in zip(vals, aligns)]}


def caprow(text, ncols, sz=1150):
    cells = [cell(text, sz=sz, bold=False, color=NAVY, align="l", fill=NOTE_FILL, gridspan=ncols)]
    cells += [cell("", hmerge=True) for _ in range(ncols - 1)]
    return {"h": 360000, "cells": cells}


# ---------------- TABLE 1: Main Results (slide6) ----------------
w1 = [3450000, 1780000, 1780000, 1780000]
rows1 = [hrow(["Method", "SLA breach", "Composite cost", "Mean tardiness"], w1)]
main = [
    ("DAHS (ours)", "1.33%", "3.09", "0.53", True),
    ("greedy-MPC (analytic lookahead)", "3.13%", "9.19", "1.82", False),
    ("Snapshot ranker (τ = 1)", "3.73%", "8.77", "1.59", False),
    ("PPO — deep RL (fair budget)", "3.85%", "3.92", "0.26", False),
    ("FIFO (best static rule)", "6.60%", "7.57", "0.62", False),
    ("Offline-RL — fitted-Q", "7.18%", "7.46", "0.53", False),
    ("PPO — deep RL (60× budget)", "11.81%", "12.60", "1.00", False),
]
for name, a, b, c, win in main:
    rows1.append(drow([name, a, b, c], ["l", "ctr", "ctr", "ctr"], sz=1200,
                      fill=WIN_FILL if win else None, bold=win, h=330000))
rows1.append(caprow("Takeaway: lower is better. DAHS has the fewest SLA breaches and the lowest cost — "
                    "2.4 points below the next-best learned method.", 4))
s6 = make_slide("Main Results — DAHS vs All Baselines", 2600, w1, rows1, None, idbase=400)

# ---------------- TABLE 2: Beating the Papers (slide16) ----------------
w2 = [2620000, 1280000, 1180000, 3520000]
rows2 = [hrow(["Published method", "Its breach", "DAHS", "Why DAHS wins"], w2)]
papers = [
    ("Offline-LD — offline RL, 2025 (closest competitor)", "7.18%", "1.33%",
     "+5.85 pts better (95% CI excludes 0); the paper's method collapses to 61.9% breach under heavy load", 520000),
    ("PPO — deep RL, 2017 (fair budget)", "3.85%", "1.33%",
     "2.9× fewer breaches; PPO never learns a state-aware policy on this problem", 470000),
    ("PPO — same, 60× budget (500k steps)", "11.81%", "1.33%",
     "More training made it worse — it collapsed to always picking one rule (FEFO)", 470000),
]
for name, a, b, why, h in papers:
    rows2.append({"h": h, "cells": [
        cell(name, sz=1200, align="l"),
        cell(a, sz=1200, align="ctr"),
        cell(b, sz=1200, align="ctr", bold=True, fill=WIN_FILL),
        cell(why, sz=1150, align="l"),
    ]})
rows2.append(caprow("Takeaway: same data, same model family, fair tuning — DAHS wins the metric that "
                    "matters most: getting orders out on time.", 4))
s16 = make_slide("Beating the Published Papers", 3000, w2, rows2, None, idbase=500)

# ---------------- TABLE 3: Model vs Model (slide17) ----------------
w3 = [2300000, 3200000, 3200000]
rows3 = [hrow(["Aspect", "The papers (Offline-LD / PPO)", "DAHS (ours)"], w3)]
mvm = [
    ("Training signal", "One bootstrapped value guess per step", "Directly measured cost of every rule"),
    ("Counterfactuals", "Never sees the rules it did not pick", "Measures the cost of all four rules"),
    ("Data needed", "Hundreds of shifts, still improving", "~25 shifts — already saturated"),
    ("Stability (25 shifts)", "Unstable — breach std 4.5 pts", "Stable — breach std 0.3 pts"),
    ("Heavy-load behaviour", "Offline-RL collapses to 61.9% breach", "Degrades gracefully to 19.4% breach"),
    ("Transparency", "Opaque action output", "Emits a named rule (FIFO/FEFO/WSPT/ATC)"),
    ("Runtime cost", "Value network evaluation", "One fast prediction — no live simulation"),
]
for asp, them, us in mvm:
    rows3.append({"h": 360000, "cells": [
        cell(asp, sz=1200, bold=True, align="l"),
        cell(them, sz=1200, align="l"),
        cell(us, sz=1200, align="l", fill=WIN_FILL),
    ]})
rows3.append(caprow("Takeaway: the difference is the training signal — a directly measured answer, "
                    "not a guessed value.", 3))
s17 = make_slide("Our Model vs the Papers' Models", 3000, w3, rows3, None, idbase=600)

# ---------------- TABLE 4: Industry (slide18) ----------------
w4 = [1850000, 3450000, 3400000]
rows4 = [hrow(["Area", "Typical industry practice", "DAHS (ours)"], w4)]
ind = [
    ("On the floor", "Fixed priority rules (e.g. FIFO / earliest-deadline) — simple, but cannot adapt as conditions change",
     "Re-picks the best rule every 15 minutes for the current state", 520000),
    ("Big-tech AI", "Deep-RL / large optimization on massive data + compute — powerful, but data-hungry, costly and opaque",
     "Big-tech-style adaptivity from ~25 shifts; one cheap, fast prediction", 520000),
    ("Transparency", "End-to-end models output assignments that are hard for staff to check",
     "Outputs a named rule a supervisor already understands and can verify", 470000),
    ("Adapting to change", "Often needs costly retraining when demand patterns shift",
     "Holds up across untuned settings and bursty real data — no retraining", 470000),
]
for area, them, us, h in ind:
    rows4.append({"h": h, "cells": [
        cell(area, sz=1200, bold=True, align="l"),
        cell(them, sz=1150, align="l"),
        cell(us, sz=1150, align="l", fill=WIN_FILL),
    ]})
rows4.append(caprow("*General, publicly-described approaches. DAHS delivers adaptivity without big-tech-scale "
                    "data, compute or black-box risk — ideal where those resources are not available.", 3))
s18 = make_slide("Industry Practice (Amazon, Flipkart) vs DAHS", 2600, w4, rows4, None, idbase=700)

for fn, xml in [("slide6.xml", s6), ("slide16.xml", s16), ("slide17.xml", s17), ("slide18.xml", s18)]:
    with open(os.path.join(SLIDES_DIR, fn), "w", encoding="utf-8") as f:
        f.write(xml)
    print("wrote", fn, len(xml), "bytes")
print("DONE")
