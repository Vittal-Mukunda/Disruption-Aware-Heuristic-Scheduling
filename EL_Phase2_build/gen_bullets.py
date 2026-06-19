# -*- coding: utf-8 -*-
"""Regenerate the 6 Phase-II bullet slides, mirroring template slide4 exactly:
red Bookman centred title, body placeholder (inherits master font) with navy
filled bullets, justified text, bold inline lead-labels. Footer date + slide num."""
import os

SLIDES_DIR = r"C:\Users\vitta\OneDrive\Desktop\DAHS\EL_Phase2_build\unpacked\ppt\slides"
DATE = "Tuesday, 02 June 2026"
RED = "C00000"
NAVY = "002060"
FONT = "Bookman Old Style"

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


def bullet(runs, sz, first):
    spcbef = "1200" if first else "0"
    pPr = ('<a:pPr indent="-349250" lvl="0" marL="457200" rtl="0" algn="just">'
           '<a:lnSpc><a:spcPct val="100000"/></a:lnSpc>'
           '<a:spcBef><a:spcPts val="%s"/></a:spcBef><a:spcAft><a:spcPts val="0"/></a:spcAft>'
           '<a:buClr><a:srgbClr val="%s"/></a:buClr><a:buSzPts val="%d"/><a:buChar char="●"/></a:pPr>'
           % (spcbef, NAVY, sz))
    rxml = ""
    for text, bold in runs:
        b = ' b="1"' if bold else ""
        rxml += '<a:r><a:rPr%s lang="en-GB" sz="%d"/><a:t>%s</a:t></a:r>' % (b, sz, esc(text))
    return '<a:p>' + pPr + rxml + ('<a:endParaRPr sz="%d"/>' % sz) + '</a:p>'


def make(title, title_sz, bullets, body_sz, idbase=300):
    body_paras = "".join(bullet(runs, body_sz, i == 0) for i, runs in enumerate(bullets))
    body = ('<p:sp><p:nvSpPr><p:cNvPr id="%d" name="Body"/><p:cNvSpPr txBox="1"/>'
            '<p:nvPr><p:ph idx="1" type="body"/></p:nvPr></p:nvSpPr>'
            '<p:spPr><a:xfrm><a:off x="131100" y="1078352"/><a:ext cx="8881800" cy="3300000"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>'
            '<p:txBody><a:bodyPr anchorCtr="0" anchor="t" bIns="34275" lIns="68575" spcFirstLastPara="1" '
            'rIns="68575" wrap="square" tIns="34275"><a:noAutofit/></a:bodyPr><a:lstStyle/>%s</p:txBody></p:sp>'
            % (idbase + 1, body_paras))

    titlebox = ('<p:sp><p:nvSpPr><p:cNvPr id="%d" name="Title"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
                '<p:spPr><a:xfrm><a:off x="372000" y="109252"/><a:ext cx="8400000" cy="629551"/></a:xfrm>'
                '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/>'
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
                % (idbase + 2, title_sz, RED, FONT, FONT, FONT, FONT, esc(title)))

    sldnum = ('<p:sp><p:nvSpPr><p:cNvPr id="%d" name="num"/><p:cNvSpPr txBox="1"/>'
              '<p:nvPr><p:ph idx="12" type="sldNum"/></p:nvPr></p:nvSpPr>'
              '<p:spPr><a:xfrm><a:off x="6955491" y="4760404"/><a:ext cx="2057400" cy="273844"/></a:xfrm>'
              '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>'
              '<p:txBody><a:bodyPr anchorCtr="0" anchor="ctr" bIns="34275" lIns="68575" spcFirstLastPara="1" '
              'rIns="68575" wrap="square" tIns="34275"><a:noAutofit/></a:bodyPr><a:lstStyle/>'
              '<a:p><a:pPr indent="0" lvl="0" marL="0" rtl="0" algn="r"><a:spcBef><a:spcPts val="0"/></a:spcBef>'
              '<a:spcAft><a:spcPts val="0"/></a:spcAft><a:buNone/></a:pPr>'
              '<a:r><a:rPr lang="en-GB"/><a:t>Slide No. </a:t></a:r>'
              '<a:fld id="{00000000-1234-1234-1234-123412341234}" type="slidenum"><a:rPr lang="en-GB"/>'
              '<a:t>‹#›</a:t></a:fld><a:endParaRPr/></a:p></p:txBody></p:sp>' % (idbase + 3))

    datebox = ('<p:sp><p:nvSpPr><p:cNvPr id="%d" name="dt"/><p:cNvSpPr txBox="1"/>'
               '<p:nvPr><p:ph idx="10" type="dt"/></p:nvPr></p:nvSpPr>'
               '<p:spPr><a:xfrm><a:off x="131109" y="4760404"/><a:ext cx="2057400" cy="273900"/></a:xfrm>'
               '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>'
               '<p:txBody><a:bodyPr anchorCtr="0" anchor="ctr" bIns="34275" lIns="68575" spcFirstLastPara="1" '
               'rIns="68575" wrap="square" tIns="34275"><a:noAutofit/></a:bodyPr><a:lstStyle/>'
               '<a:p><a:pPr indent="0" lvl="0" marL="0" rtl="0" algn="ctr"><a:spcBef><a:spcPts val="0"/></a:spcBef>'
               '<a:spcAft><a:spcPts val="0"/></a:spcAft><a:buNone/></a:pPr>'
               '<a:r><a:rPr lang="en-GB"/><a:t>%s</a:t></a:r><a:endParaRPr/></a:p></p:txBody></p:sp>' % (idbase + 4, DATE))

    tree = ('<p:nvGrpSpPr><p:cNvPr id="%d" name="Shape"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
            '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
            '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
            % idbase) + body + titlebox + sldnum + datebox
    return (HEADER + '<p:cSld><p:spTree>' + tree + '</p:spTree></p:cSld>'
            '<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>')


B = lambda t: (t, True)   # bold run
N = lambda t: (t, False)  # normal run

# slide4 — Recap of Phase-I
s4 = make("Recap of Phase-I", 3000, [
    [B("Problem: "), N("e-commerce warehouses must dispatch orders under constant change — demand surges, perishable goods and tight delivery deadlines (SLAs). One fixed rule cannot stay best all day.")],
    [B("Phase-I proposal: "), N("DAHS — a machine-learning scheduler that picks the best dispatching rule for the current situation, instead of using one rule for the whole shift.")],
    [B("The plan: "), N("build a realistic warehouse simulator, capture disruption-aware features, and train a model to choose the right rule at each decision point.")],
    [B("The goal: "), N("fewer late orders (SLA breaches) and lower delays (tardiness), without the cost and instability of heavy AI methods.")],
    [B("Phase-II task: "), N("actually build it, test it against strong baselines and published papers, and prove that it works.")],
], 1800, idbase=300)

# slide5 — Phase-II Objectives
s5 = make("Phase-II Objectives", 3000, [
    [B("Build the full system: "), N("a complete warehouse simulator, the training pipeline and the final DAHS model, working end-to-end.")],
    [B("Refine the method: "), N("replace the early Random-Forest idea with a stronger approach — offline rollout distillation feeding a calibrated gradient-boosted ranker.")],
    [B("Compare fairly: "), N("test against static rules, an analytic lookahead controller, a contextual bandit, and two published learning methods (PPO and offline-RL).")],
    [B("Prove it, do not just claim it: "), N("measure SLA breach, cost and tardiness on 50 unseen shifts, with confidence intervals and significance tests.")],
    [B("Stress-test: "), N("check robustness across many untuned settings and under realistic, bursty real-world order data.")],
], 1800, idbase=320)

# slide7 — Core Idea
s7 = make("How DAHS Works — The Core Idea", 2400, [
    [B("Four simple rules in the toolbox: "), N("FIFO (oldest first), FEFO (earliest-expiry first), WSPT (shortest job first), ATC (urgency-weighted). Each is fast and easy to understand.")],
    [B("No single rule wins: "), N("the best rule keeps changing during a shift — FEFO is best 43% of the time, WSPT 32%, FIFO 15%, ATC 10%. So which rule to use is a real decision.")],
    [B("DAHS is a selector: "), N("every 15 minutes it reads the warehouse state and picks the rule expected to cost the least over the next hour.")],
    [B("Learned cheaply, offline: "), N("we simulate each rule's short-term outcome once, ahead of time, and teach a model to copy the best choice.")],
    [B("Instant at runtime: "), N("one fast prediction, no live simulation — and the chosen rule is a name a supervisor can see and trust.")],
], 1800, idbase=340)

# slide10 — Pipeline
s10 = make("How DAHS Works — The 4-Step Pipeline", 2400, [
    [B("Step 1 — Gather experience: "), N("run 250 simulated 8-hour shifts trying all rules, collecting 8,000 real decision moments.")],
    [B("Step 2 — Measure the truth (rollout): "), N("at each moment, briefly simulate every rule forward and record its actual cost — the direct answer, not a guess.")],
    [B("Step 3 — Find operating regimes: "), N("automatically group situations (quiet start, busy peak, perishable rush) so the model knows the context.")],
    [B("Step 4 — Train a calibrated ranker: "), N("a gradient-boosted model learns to predict the best rule; its confidence is calibrated so a light switching controller can act safely.")],
    [B("Deploy: "), N("one forward pass per 15-minute interval picks the rule — transparent, fast and stable.")],
], 1800, idbase=360)

# slide11 — Experimental Setup
s11 = make("Experimental Setup", 3000, [
    [B("Simulator: "), N("an 8-hour shift split into 32 intervals of 15 minutes, 10 pickers, orders arriving randomly, 20% perishable, each with a deadline.")],
    [B("Fair testing: "), N("all methods judged on the same 50 unseen shifts; training used a separate 250 shifts.")],
    [B("Scenarios: "), N("light load, balanced, default, and a tough high-load-perishable case — all fixed in advance, never tuned per method.")],
    [B("Baselines: "), N("the 4 static rules, a snapshot model, an analytic one-step lookahead (greedy-MPC), a contextual bandit (LinUCB), PPO (deep RL) and offline-RL (fitted-Q).")],
    [B("Metrics: "), N("SLA-breach rate (main), composite cost, tardiness, throughput and picker use — with bootstrap confidence intervals and significance tests.")],
], 1700, idbase=380)

# slide12 — Conclusion
s12 = make("Conclusion", 3000, [
    [B("DAHS works: "), N("1.33% SLA-breach vs 3.13% (analytic) and 7.18% (offline-RL) — the fewest breaches and the lowest cost on unseen shifts.")],
    [B("Headline = sample efficiency: "), N("a deployable controller from just 25 shifts, beating methods trained on 10x more data.")],
    [B("Beats published methods: "), N("outperforms PPO and an offline-RL (Offline-LD style) method by wide, statistically significant margins.")],
    [B("Robust and realistic: "), N("Pareto non-dominated, stable across 12 untuned settings, and its lead widens under real bursty order data.")],
    [B("Practical edge: "), N("transparent named rules, instant decisions, no live simulation — adaptivity without heavy-AI cost.")],
    [B("Next: "), N("validate on a real warehouse log and a larger rule pool. Phase-II is complete and ready to finish.")],
], 1550, idbase=400)

for fn, xml in [("slide4.xml", s4), ("slide5.xml", s5), ("slide7.xml", s7),
                ("slide10.xml", s10), ("slide11.xml", s11), ("slide12.xml", s12)]:
    with open(os.path.join(SLIDES_DIR, fn), "w", encoding="utf-8") as f:
        f.write(xml)
    print("wrote", fn, len(xml))
print("DONE")
