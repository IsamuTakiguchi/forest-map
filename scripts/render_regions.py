# -*- coding: utf-8 -*-
"""segment.py の結果(masks.npz, regions.json)を使い、森林計画図に区画を赤で囲んだPDFを作る。
使い方: python3 render_regions.py 入力.pdf 出力.pdf 出力1頁目.png  (カレントに masks.npz, regions.json が必要)"""
import sys, json, numpy as np, cv2, pymupdf
from PIL import Image, ImageDraw, ImageFont
SRC, DST, PNG = sys.argv[1], sys.argv[2], sys.argv[3]
FONT = '/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf'
RED = (220, 0, 0)
sys.path.insert(0, __import__('os').path.dirname(__file__))
src = open(__import__('os').path.join(__import__('os').path.dirname(__file__), 'annotate.py')).read()
ns = {}; exec(src[src.index('LOTS = ['):src.index('NOT_FOUND')], ns); LOTS = ns['LOTS']
NOT_FOUND = {'オ': [2, 5, 11]}
APPROX_FIXED = {'ワ1': '図面上端で切れているため北側が不完全'}

doc = pymupdf.open(SRC); p0 = doc[0]
xref = p0.get_images()[0][0]; pix = pymupdf.Pixmap(doc, xref)
gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
H, W = gray.shape
img = np.stack([gray]*3, -1).copy()
regions = json.load(open('regions.json')); mz = np.load('masks.npz')
APPROX = dict(APPROX_FIXED)
for k, r in regions.items():
    if r.get('manual'): APPROX[k] = '番号周辺の区画線から手動で推定'
    elif not r.get('ok'): APPROX[k] = '区画線の判読が不確実なため概略'
# 領域の重なりは小さい区画を優先(大きい区画から小さい区画を除く)
raw = {}
for sub,no,*_ in LOTS:
    key = f'{sub}{no}'
    raw[key] = np.unpackbits(mz[key])[:H*W].reshape(H,W).astype(bool)
keys_by_size = sorted(raw, key=lambda k: raw[k].sum())
for i, small in enumerate(keys_by_size):
    for big in keys_by_size[i+1:]:
        if (raw[small] & raw[big]).any(): raw[big] &= ~raw[small]

# 1) 半透明の赤塗り
overlay = img.copy()
polys = {}
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
for sub,no,*_ in LOTS:
    key = f'{sub}{no}'
    m = raw[key].astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))   # 細い突起を除去
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))  # 等高線による細い切れ込みを埋める
    m = cv2.dilate(m, k5)                    # 区画線の太らせ分を戻す
    cs,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = [c for c in cs if cv2.contourArea(c) > 150]
    polys[key] = [cv2.approxPolyDP(c, 1.5, True) for c in cs]
    cv2.fillPoly(overlay, polys[key], (255, 120, 120))
img = cv2.addWeighted(overlay, 0.28, img, 0.72, 0)
# 2) 赤の輪郭線
for key, cs in polys.items():
    cv2.polylines(img, cs, True, RED, 5, cv2.LINE_AA)

pil = Image.fromarray(img); d = ImageDraw.Draw(pil)
f_tag = ImageFont.truetype(FONT, 27)

def clip_to_box(cx, cy, bx, by, box):
    x0,y0,x1,y1 = box
    for t in [i/200 for i in range(201)]:
        x = cx+(bx-cx)*t; y = cy+(by-cy)*t
        if x0<=x<=x1 and y0<=y<=y1: return x, y
    return bx, by
def tag(text, box, side, font, pad=4):
    x0,y0,x1,y1 = box
    tw = d.textlength(text, font=font); th = font.size
    if isinstance(side, tuple):
        tx, ty = side
        tcx, tcy = tx+(tw+2*pad)/2, ty+(th+2*pad)/2
        ex, ey = clip_to_box(tcx, tcy, (x0+x1)/2, (y0+y1)/2, box)
        d.line([(tcx,tcy),(ex,ey)], fill=RED, width=3)
    elif side == 'r': tx, ty = x1+6, (y0+y1)/2 - th/2
    elif side == 'l': tx, ty = x0-6-tw-2*pad, (y0+y1)/2 - th/2
    elif side == 't': tx, ty = (x0+x1)/2 - tw/2, y0-6-th-2*pad
    else:             tx, ty = (x0+x1)/2 - tw/2, y1+6
    d.rectangle([tx, ty, tx+tw+2*pad, ty+th+2*pad], fill=(255,255,255), outline=RED, width=2)
    d.text((tx+pad, ty+pad-2), text, font=font, fill=RED)
for sub,no,x0,y0,x1,y1,side in LOTS:
    key=f'{sub}{no}'
    tag(key + ('※' if key in APPROX else ''), (x0,y0,x1,y1), side, f_tag)

# 凡例
LEG_H = 400
canvas = Image.new('RGB', (W, H+LEG_H), (255,255,255)); canvas.paste(pil, (0,0))
d = ImageDraw.Draw(canvas); d.line([(0,H),(W,H)], fill=RED, width=4)
f_h = ImageFont.truetype(FONT, 44); f_b = ImageFont.truetype(FONT, 36)
y = H+18
d.rectangle([60,y+4,120,y+44], fill=(255,215,215), outline=RED, width=5)
d.text((140,y), '赤枠（薄赤塗り）＝ 森林簿(2〜4頁)に記載された施業番号(林班57)の区画の範囲。「ル9」等のタグは 小班名＋施業番号。', font=f_h, fill=RED)
y += 66
groups = {}
for sub,no,*_ in LOTS: groups.setdefault(sub, []).append(no)
for sub in ['ル','オ','ワ','カ']:
    s = f'小班{sub}：' + ', '.join(str(n) for n in groups[sub])
    if sub in NOT_FOUND: s += f'　（※{ "・".join(map(str,NOT_FOUND[sub])) } は図面上に番号の表示を確認できず）'
    d.text((70,y), s, font=f_b, fill=(0,0,0)); y += 46
apx = {}
for k,v in APPROX.items(): apx.setdefault(v, []).append(k)
d.text((70,y+4), '※印：' + '、'.join(f'{"・".join(ks)}は{v}' for v,ks in apx.items()) + '。', font=f_b, fill=(80,80,80)); y += 46
d.text((70,y+4), '区画は図面の直線的な区画線（等高線は除外）を画像処理で抽出し、森林簿の面積と照合して求めた。境界の細部は原図で確認のこと。', font=f_b, fill=(80,80,80))
canvas.save(PNG, dpi=(300,300))

out = pymupdf.open()
pw, ph = p0.rect.width, p0.rect.width * canvas.height / canvas.width
page = out.new_page(width=pw, height=ph); page.insert_image(page.rect, filename=PNG)
out.insert_pdf(doc, from_page=1, to_page=len(doc)-1)
out.save(DST, garbage=3, deflate=True)
print('saved', DST, len(out), 'pages')
