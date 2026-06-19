# -*- coding: utf-8 -*-
"""Write the References slide (slide13) and globally update the footer date."""
import os, glob

SLIDES = r"C:\Users\vitta\OneDrive\Desktop\DAHS\EL_Phase2_build\unpacked\ppt\slides"
OLD_DATE = "Tuesday, 07 April 2026"
NEW_DATE = "Tuesday, 02 June 2026"
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


REFS = [
    "[1] Pluijm et al. (2025). Offline-LD: Offline Reinforcement Learning with maskable Q-learning for scheduling. arXiv. — the closest published competitor (our offline-RL baseline).",
    "[2] Schulman, Wolski, Dhariwal, Radford & Klimov (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.",
    "[3] Bertsekas (2020). Rollout, Policy Iteration, and Distributed Reinforcement Learning. Athena Scientific.",
    "[4] Ernst, Geurts & Wehenkel (2005). Tree-Based Batch Mode Reinforcement Learning (fitted Q-iteration). JMLR 6, 503–556.",
    "[5] Drake, Kheiri, Özcan & Burke (2020). Recent advances in selection hyper-heuristics. European Journal of Operational Research 285(2).",
    "[6] Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. ACM SIGKDD.",
    "[7] Mahmoudinazlou et al. (2025). Deep reinforcement learning for dynamic order picking in warehouses. Computers & Operations Research 182.",
    "[8] Olist (2018). Brazilian E-Commerce Public Dataset. Kaggle.",
]


def ref_para(text):
    return ('<a:p><a:pPr indent="0" lvl="0" marL="0" rtl="0" algn="just">'
            '<a:lnSpc><a:spcPct val="100000"/></a:lnSpc>'
            '<a:spcBef><a:spcPts val="450"/></a:spcBef><a:spcAft><a:spcPts val="0"/></a:spcAft>'
            '<a:buNone/></a:pPr><a:r><a:rPr lang="en-GB" sz="1300"/><a:t>%s</a:t></a:r>'
            '<a:endParaRPr sz="1300"/></a:p>' % esc(text))


def build_refs():
    body = ('<p:sp><p:nvSpPr><p:cNvPr id="273" name="Body"/><p:cNvSpPr txBox="1"/>'
            '<p:nvPr><p:ph idx="1" type="body"/></p:nvPr></p:nvSpPr>'
            '<p:spPr><a:xfrm><a:off x="292100" y="820000"/><a:ext cx="8560000" cy="3900000"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>'
            '<p:txBody><a:bodyPr anchorCtr="0" anchor="t" bIns="34275" lIns="68575" spcFirstLastPara="1" '
            'rIns="68575" wrap="square" tIns="34275"><a:noAutofit/></a:bodyPr><a:lstStyle/>%s</p:txBody></p:sp>'
            % "".join(ref_para(r) for r in REFS))

    title = ('<p:sp><p:nvSpPr><p:cNvPr id="275" name="Title"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
             '<p:spPr><a:xfrm><a:off x="1713271" y="109252"/><a:ext cx="6109800" cy="629700"/></a:xfrm>'
             '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:solidFill><a:schemeClr val="lt1"/></a:solidFill>'
             '<a:ln cap="flat" cmpd="sng" w="9525"><a:solidFill><a:schemeClr val="lt1"/></a:solidFill>'
             '<a:prstDash val="dash"/><a:miter lim="800000"/>'
             '<a:headEnd len="sm" w="sm" type="none"/><a:tailEnd len="sm" w="sm" type="none"/></a:ln></p:spPr>'
             '<p:txBody><a:bodyPr anchorCtr="0" anchor="ctr" bIns="34275" lIns="68575" spcFirstLastPara="1" '
             'rIns="68575" wrap="square" tIns="34275"><a:noAutofit/></a:bodyPr><a:lstStyle/>'
             '<a:p><a:pPr indent="0" lvl="0" marL="0" marR="0" rtl="0" algn="ctr">'
             '<a:spcBef><a:spcPts val="0"/></a:spcBef><a:spcAft><a:spcPts val="0"/></a:spcAft><a:buNone/></a:pPr>'
             '<a:r><a:rPr b="1" lang="en-GB" sz="3000"><a:solidFill><a:srgbClr val="%s"/></a:solidFill>'
             '<a:latin typeface="%s"/><a:ea typeface="%s"/><a:cs typeface="%s"/><a:sym typeface="%s"/></a:rPr>'
             '<a:t>References</a:t></a:r></a:p></p:txBody></p:sp>' % (RED, FONT, FONT, FONT, FONT))

    sldnum = ('<p:sp><p:nvSpPr><p:cNvPr id="274" name="num"/><p:cNvSpPr txBox="1"/>'
              '<p:nvPr><p:ph idx="12" type="sldNum"/></p:nvPr></p:nvSpPr>'
              '<p:spPr><a:xfrm><a:off x="6955491" y="4760404"/><a:ext cx="2057400" cy="273900"/></a:xfrm>'
              '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>'
              '<p:txBody><a:bodyPr anchorCtr="0" anchor="ctr" bIns="34275" lIns="68575" spcFirstLastPara="1" '
              'rIns="68575" wrap="square" tIns="34275"><a:noAutofit/></a:bodyPr><a:lstStyle/>'
              '<a:p><a:pPr indent="0" lvl="0" marL="0" rtl="0" algn="r"><a:spcBef><a:spcPts val="0"/></a:spcBef>'
              '<a:spcAft><a:spcPts val="0"/></a:spcAft><a:buNone/></a:pPr>'
              '<a:fld id="{00000000-1234-1234-1234-123412341234}" type="slidenum"><a:rPr lang="en-GB"/>'
              '<a:t>‹#›</a:t></a:fld><a:endParaRPr/></a:p></p:txBody></p:sp>')

    datebox = ('<p:sp><p:nvSpPr><p:cNvPr id="276" name="dt"/><p:cNvSpPr txBox="1"/>'
               '<p:nvPr><p:ph idx="10" type="dt"/></p:nvPr></p:nvSpPr>'
               '<p:spPr><a:xfrm><a:off x="131109" y="4760404"/><a:ext cx="2057400" cy="273900"/></a:xfrm>'
               '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>'
               '<p:txBody><a:bodyPr anchorCtr="0" anchor="ctr" bIns="34275" lIns="68575" spcFirstLastPara="1" '
               'rIns="68575" wrap="square" tIns="34275"><a:noAutofit/></a:bodyPr><a:lstStyle/>'
               '<a:p><a:pPr indent="0" lvl="0" marL="0" rtl="0" algn="ctr"><a:spcBef><a:spcPts val="0"/></a:spcBef>'
               '<a:spcAft><a:spcPts val="0"/></a:spcAft><a:buNone/></a:pPr>'
               '<a:r><a:rPr lang="en-GB"/><a:t>%s</a:t></a:r><a:endParaRPr/></a:p></p:txBody></p:sp>' % NEW_DATE)

    tree = ('<p:nvGrpSpPr><p:cNvPr id="272" name="Shape"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
            '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
            '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>') + body + title + sldnum + datebox
    return (HEADER + '<p:cSld><p:spTree>' + tree + '</p:spTree></p:cSld>'
            '<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>')


with open(os.path.join(SLIDES, "slide13.xml"), "w", encoding="utf-8") as f:
    f.write(build_refs())
print("wrote slide13.xml (references)")

# global date update
n = 0
for fn in glob.glob(os.path.join(SLIDES, "slide*.xml")):
    with open(fn, "r", encoding="utf-8") as f:
        s = f.read()
    if OLD_DATE in s:
        with open(fn, "w", encoding="utf-8") as f:
            f.write(s.replace(OLD_DATE, NEW_DATE))
        n += 1
print("updated date in", n, "slides")
