# -*- coding: utf-8 -*-
"""森林計画図(1頁目)に森林簿(2頁目以降)記載の施業番号の位置を赤枠で表示する。"""
import sys, pymupdf
from PIL import Image, ImageDraw, ImageFont

SRC, DST, PNG = sys.argv[1], sys.argv[2], sys.argv[3]
FONT = '/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf'
RED = (220, 0, 0)

# (小班, 番号, x0, y0, x1, y1, タグ位置)  ※元画像(3310x2340px)座標
LOTS = [
 # ---- ル ----
 ('ル', 9,  1566,1733,1602,1770,'l'),
 ('ル', 15, 1620,1590,1680,1642,'l'),
 ('ル', 16, 1524,1154,1586,1206,'l'),
 ('ル', 17, 1540,1508,1574,1542,'r'),
 ('ル', 19, 1666,1482,1700,1516,'t'),
 ('ル', 20, 1636,1016,1686,1060,'t'),
 ('ル', 21, 1510,1353,1556,1392,'l'),
 ('ル', 22, 1546,1034,1647,1068,'b'),
 # ---- オ ----
 ('オ', 1,  1603,756,1642,792,'r'),
 ('オ', 3,  1674,646,1704,678,'r'),
 ('オ', 4,  1560,551,1600,595,'l'),
 ('オ', 6,  1686,558,1718,595,'l'),
 ('オ', 7,  1530,456,1569,502,'l'),
 ('オ', 8,  1710,508,1744,545,'l'),
 ('オ', 9,  1662,465,1703,509,'t'),
 ('オ', 10, 1578,371,1629,412,'l'),
 ('オ', 12, 1643,340,1692,385,'b'),
 ('オ', 13, 1720,314,1768,354,'r'),
 ('オ', 14, 1646,238,1692,277,'l'),
 ('オ', 15, 1740,228,1782,264,'r'),
 ('オ', 16, 1693,147,1737,190,'l'),
 ('オ', 17, 1703,47,1747,90,'l'),
 ('オ', 18, 1531,753,1594,802,'b'),
 ('オ', 19, 1575,618,1622,665,'l'),
 ('オ', 20, 1672,616,1707,648,'r'),
 # ---- ワ ----
 ('ワ', 1,  1795,88,1828,124,'r'),
 ('ワ', 5,  1775,476,1812,514,'t'),
 ('ワ', 6,  1900,426,1937,464,'r'),
 ('ワ', 7,  1712,562,1747,602,'r'),
 ('ワ', 8,  1806,558,1845,600,'r'),
 ('ワ', 9,  1695,706,1732,747,(1622,712)),
 ('ワ', 10, 1733,706,1782,747,'b'),
 ('ワ', 11, 1798,681,1847,724,'r'),
 ('ワ', 12, 1682,794,1720,822,(1622,796)),
 ('ワ', 13, 1730,789,1770,820,'b'),
 ('ワ', 14, 1684,828,1721,857,(1622,832)),
 ('ワ', 15, 1796,790,1842,828,'r'),
 ('ワ', 16, 1668,894,1706,926,'l'),
 ('ワ', 17, 1748,920,1797,960,'r'),
 ('ワ', 18, 1674,1039,1707,1068,'r'),
 # ---- カ ----
 ('カ', 1,  1690,1113,1717,1152,'r'),
 ('カ', 2,  1681,1670,1720,1717,'l'),
 ('カ', 3,  1776,1178,1812,1222,'l'),
 ('カ', 4,  1814,1208,1842,1248,'b'),
 ('カ', 5,  1736,1648,1768,1692,'b'),
]
# 小班名ラベル(カタカナ)の位置
SUBS = {
 'ル': (1497,1470,1548,1560),
 'オ': (1598,438,1652,515),
 'ワ': (1804,470,1848,532),
 'カ': (1713,1522,1752,1592),
}
NOT_FOUND = {'オ': [2, 5, 11]}

doc = pymupdf.open(SRC)
p0 = doc[0]
xref = p0.get_images()[0][0]
pix = pymupdf.Pixmap(doc, xref)
img = Image.frombytes('L', (pix.width, pix.height), pix.samples).convert('RGB')
W, H = img.size
d = ImageDraw.Draw(img)
f_tag = ImageFont.truetype(FONT, 27)
f_sub = ImageFont.truetype(FONT, 32)

def clip_to_box(cx, cy, bx, by, box):
    """(cx,cy)->(bx,by) の線分が矩形boxに入る点を返す"""
    x0,y0,x1,y1 = box
    t_best = 1.0
    for t in [i/200 for i in range(201)]:
        x = cx+(bx-cx)*t; y = cy+(by-cy)*t
        if x0<=x<=x1 and y0<=y<=y1:
            t_best = t; break
    return cx+(bx-cx)*t_best, cy+(by-cy)*t_best

def tag(text, box, side, font, pad=4):
    x0,y0,x1,y1 = box
    tw = d.textlength(text, font=font); th = font.size
    if isinstance(side, tuple):
        tx, ty = side
        # 引出線
        tcx, tcy = tx+(tw+2*pad)/2, ty+(th+2*pad)/2
        ex, ey = clip_to_box(tcx, tcy, (x0+x1)/2, (y0+y1)/2, box)
        d.line([(tcx,tcy),(ex,ey)], fill=RED, width=3)
    elif side == 'r':   tx, ty = x1+6, (y0+y1)/2 - th/2
    elif side == 'l': tx, ty = x0-6-tw-2*pad, (y0+y1)/2 - th/2
    elif side == 't': tx, ty = (x0+x1)/2 - tw/2, y0-6-th-2*pad
    else:             tx, ty = (x0+x1)/2 - tw/2, y1+6
    d.rectangle([tx, ty, tx+tw+2*pad, ty+th+2*pad], fill=(255,255,255), outline=RED, width=2)
    d.text((tx+pad, ty+pad-2), text, font=font, fill=RED)

SUB_TAG = {'ル':'l', 'オ':(1445,505), 'ワ':'r', 'カ':'r'}
for name,(x0,y0,x1,y1) in SUBS.items():
    d.rounded_rectangle([x0-8,y0-8,x1+8,y1+8], radius=10, outline=RED, width=7)
    tag(f'小班 {name}', (x0-8,y0-8,x1+8,y1+8), SUB_TAG[name], f_sub, pad=6)

for sub,no,x0,y0,x1,y1,side in LOTS:
    d.rectangle([x0,y0,x1,y1], outline=RED, width=5)
    tag(f'{sub}{no}', (x0,y0,x1,y1), side, f_tag)

# 凡例(画像下部に白帯を追加)
LEG_H = 340
canvas = Image.new('RGB', (W, H+LEG_H), (255,255,255))
canvas.paste(img, (0,0))
d = ImageDraw.Draw(canvas)
d.line([(0,H),(W,H)], fill=RED, width=4)
f_h = ImageFont.truetype(FONT, 44); f_b = ImageFont.truetype(FONT, 36)
y = H+18
d.rectangle([60,y+4,120,y+44], outline=RED, width=5)
d.text((140,y), '赤枠 ＝ 森林簿(2〜4頁)に記載された施業番号(林班57)の図面上の位置。枠の横の「ル9」等は 小班名＋施業番号。', font=f_h, fill=RED)
y += 66
groups = {}
for sub,no,*_ in LOTS: groups.setdefault(sub, []).append(no)
lines = []
for sub in ['ル','オ','ワ','カ']:
    nos = ', '.join(str(n) for n in groups[sub])
    s = f'小班{sub}：{nos}'
    if sub in NOT_FOUND:
        s += f'　（※{ "・".join(map(str,NOT_FOUND[sub])) } は図面上に番号の表示を確認できず）'
    lines.append(s)
for s in lines:
    d.text((70,y), s, font=f_b, fill=(0,0,0)); y += 46
d.text((70,y+4), '※赤枠は番号ラベルの位置を示すもので、各区画の範囲は図面の区画線を参照のこと。太い赤角枠は小班名(ル・オ・ワ・カ)の位置。', font=f_b, fill=(80,80,80))

canvas.save(PNG, dpi=(300,300))

# PDF組み立て: 1頁目=注記済み画像、2頁目以降=原本のまま
out = pymupdf.open()
pw, ph = p0.rect.width, p0.rect.width * canvas.height / canvas.width
page = out.new_page(width=pw, height=ph)
page.insert_image(page.rect, filename=PNG)
out.insert_pdf(doc, from_page=1, to_page=len(doc)-1)
out.save(DST, garbage=3, deflate=True)
print('saved', DST, 'pages', len(out), 'size', pw, ph)
