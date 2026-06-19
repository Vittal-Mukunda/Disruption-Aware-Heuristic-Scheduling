# -*- coding: utf-8 -*-
"""Build the 2 figure slides (Sample Efficiency, Robustness): red Bookman title,
2-line navy takeaway caption, large full-width result figure fitted to true
aspect ratio. Copies PNGs into media and repoints each slide's rId3 image."""
import os, shutil
from PIL import Image

ROOT = r"C:\Users\vitta\OneDrive\Desktop\DAHS"
UNP = os.path.join(ROOT, "EL_Phase2_build", "unpacked")
SLIDES = os.path.join(UNP, "ppt", "slides")
MEDIA = os.path.join(UNP, "ppt", "media")
RELS = os.path.join(SLIDES, "_rels")
DATE = "Tuesday, 02 June 2026"
RED, NAVY, FONT = "C00000", "002060", "Bookman Old Style"

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


def cap_para(runs):
    rxml = ""
    for text, bold in runs:
        b = ' b="1"' if bold else ""
        rxml += ('<a:r><a:rPr%s lang="en-GB" sz="1400"><a:solidFill><a:srgbClr val="%s"/></a:solidFill>'
                 '<a:latin typeface="%s"/><a:ea typeface="%s"/><a:cs typeface="%s"/><a:sym typeface="%s"/>'
                 '</a:rPr><a:t>%s</a:t></a:r>' % (b, NAVY, FONT, FONT, FONT, FONT, esc(text)))
    return ('<a:p><a:pPr indent="0" lvl="0" marL="0" rtl="0" algn="l">'
            '<a:lnSpc><a:spcPct val="100000"/></a:lnSpc>'
            '<a:spcBef><a:spcPts val="300"/></a:spcBef><a:spcAft><a:spcPts val="0"/></a:spcAft>'
            '<a:buNone/></a:pPr>' + rxml + '</a:p>')


def make_fig_slide(title, caption_paras, img_px, idbase=800):
    iw, ih = img_px
    aspect = iw / ih
    box_w, y_top, y_bot = 8544000, 1840000, 4660000
    avail_h = y_bot - y_top
    w = box_w
    h = int(w / aspect)
    if h > avail_h:
        h = avail_h
        w = int(h * aspect)
    offx = (9144000 - w) // 2
    offy = y_top + (avail_h - h) // 2

    pic = ('<p:pic><p:nvPicPr><p:cNvPr id="%d" name="Figure"/><p:cNvPicPr preferRelativeResize="0"/>'
           '<p:nvPr/></p:nvPicPr><p:blipFill rotWithShape="1"><a:blip r:embed="rId3"><a:alphaModFix/></a:blip>'
           '<a:stretch><a:fillRect/></a:stretch></p:blipFill>'
           '<p:spPr><a:xfrm><a:off x="%d" y="%d"/><a:ext cx="%d" cy="%d"/></a:xfrm>'
           '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr></p:pic>'
           % (idbase + 5, offx, offy, w, h))

    caption = ('<p:sp><p:nvSpPr><p:cNvPr id="%d" name="Caption"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
               '<p:spPr><a:xfrm><a:off x="320000" y="755000"/><a:ext cx="8504000" cy="1040000"/></a:xfrm>'
               '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>'
               '<p:txBody><a:bodyPr anchorCtr="0" anchor="t" bIns="20000" lIns="20000" spcFirstLastPara="1" '
               'rIns="20000" wrap="square" tIns="20000"><a:noAutofit/></a:bodyPr><a:lstStyle/>%s</p:txBody></p:sp>'
               % (idbase + 6, "".join(cap_para(p) for p in caption_paras)))

    titlebox = ('<p:sp><p:nvSpPr><p:cNvPr id="%d" name="Title"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
                '<p:spPr><a:xfrm><a:off x="372000" y="67310"/><a:ext cx="8400000" cy="629551"/></a:xfrm>'
                '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/>'
                '<a:ln cap="flat" cmpd="sng" w="9525"><a:solidFill><a:schemeClr val="lt1"/></a:solidFill>'
                '<a:prstDash val="dash"/><a:miter lim="800000"/>'
                '<a:headEnd len="sm" w="sm" type="none"/><a:tailEnd len="sm" w="sm" type="none"/></a:ln></p:spPr>'
                '<p:txBody><a:bodyPr anchorCtr="0" anchor="ctr" bIns="34275" lIns="68575" spcFirstLastPara="1" '
                'rIns="68575" wrap="square" tIns="34275"><a:noAutofit/></a:bodyPr><a:lstStyle/>'
                '<a:p><a:pPr indent="0" lvl="0" marL="0" marR="0" rtl="0" algn="ctr">'
                '<a:spcBef><a:spcPts val="0"/></a:spcBef><a:spcAft><a:spcPts val="0"/></a:spcAft><a:buNone/></a:pPr>'
                '<a:r><a:rPr b="1" lang="en-GB" sz="2200"><a:solidFill><a:srgbClr val="%s"/></a:solidFill>'
                '<a:latin typeface="%s"/><a:ea typeface="%s"/><a:cs typeface="%s"/><a:sym typeface="%s"/></a:rPr>'
                '<a:t>%s</a:t></a:r></a:p></p:txBody></p:sp>'
                % (idbase + 1, RED, FONT, FONT, FONT, FONT, esc(title)))

    sldnum = ('<p:sp><p:nvSpPr><p:cNvPr id="%d" name="num"/><p:cNvSpPr txBox="1"/>'
              '<p:nvPr><p:ph idx="12" type="sldNum"/></p:nvPr></p:nvSpPr>'
              '<p:spPr><a:xfrm><a:off x="6955491" y="4760404"/><a:ext cx="2057400" cy="273844"/></a:xfrm>'
              '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>'
              '<p:txBody><a:bodyPr anchorCtr="0" anchor="ctr" bIns="34275" lIns="68575" spcFirstLastPara="1" '
              'rIns="68575" wrap="square" tIns="34275"><a:noAutofit/></a:bodyPr><a:lstStyle/>'
              '<a:p><a:pPr indent="0" lvl="0" marL="0" rtl="0" algn="r"><a:spcBef><a:spcPts val="0"/></a:spcBef>'
              '<a:spcAft><a:spcPts val="0"/></a:spcAft><a:buNone/></a:pPr>'
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
            % idbase) + titlebox + caption + pic + sldnum + datebox
    return (HEADER + '<p:cSld><p:spTree>' + tree + '</p:spTree></p:cSld>'
            '<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>')


def repoint(rels_file, target):
    with open(rels_file, "r", encoding="utf-8") as f:
        s = f.read()
    import re
    s = re.sub(r'(Id="rId3"[^>]*Target=")[^"]*(")', r'\g<1>' + target + r'\g<2>', s)
    with open(rels_file, "w", encoding="utf-8") as f:
        f.write(s)


B = lambda t: (t, True)
N = lambda t: (t, False)

# --- copy figures into media ---
se_src = os.path.join(ROOT, "figures", "data_efficiency", "data_efficiency_curve.png")
rob_src = os.path.join(ROOT, "figures", "A2", "olist_arrivals_compare.png")
shutil.copy(se_src, os.path.join(MEDIA, "image_se.png"))
shutil.copy(rob_src, os.path.join(MEDIA, "image_rob.png"))
se_px = Image.open(se_src).size
rob_px = Image.open(rob_src).size

repoint(os.path.join(RELS, "slide8.xml.rels"), "../media/image_se.png")
repoint(os.path.join(RELS, "slide9.xml.rels"), "../media/image_rob.png")

s8 = make_fig_slide("Sample Efficiency — The Key Result", [
    [B("Just 25 training shifts"), N(" (about 2.5 hours of simulation) already beat every baseline at any data budget.")],
    [N("DAHS even beats offline-RL trained on "), B("10x more data"), N(" — each training moment carries a directly measured answer, so learning is dense and stable.")],
], se_px, idbase=800)

s9 = make_fig_slide("Robustness & Real-World Validation", [
    [N("Tested on bursty arrivals from a real e-commerce trace (Olist, ~100k orders): "), B("DAHS keeps rank one on every metric.")],
    [N("Its lead actually "), B("widens under realistic bursts"), N(", and it stays Pareto-best across all 4 scenarios and 12 untuned settings.")],
], rob_px, idbase=850)

with open(os.path.join(SLIDES, "slide8.xml"), "w", encoding="utf-8") as f:
    f.write(s8)
with open(os.path.join(SLIDES, "slide9.xml"), "w", encoding="utf-8") as f:
    f.write(s9)
print("se", se_px, "rob", rob_px, "DONE")
