import numpy as np, cv2, sys, json, random, os
from PIL import Image, ImageDraw, ImageFont
T, MIN, DI = 90, 20, 5
CFG = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'segment_config.json')))
BARRIERS = CFG.get('barriers', []); ERASERS = CFG.get('erasers', []); SEEDS = CFG.get('seeds', {}); MANUAL = CFG.get('manual', {})
im = np.array(Image.open('map_full.png'))
H,W = im.shape
dark = (im < T).astype(np.uint8)
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'annotate.py')).read()
ns = {}; exec(src[src.index('LOTS = ['):src.index('NOT_FOUND')], ns); LOTS = ns['LOTS']
HA = {'ル9':0.06,'ル15':0.22,'ル16':4.3,'ル17':0.09,'ル19':0.05,'ル20':0.77,'ル21':0.72,'ル22':0.06,
 'オ1':1.2,'オ3':0.04,'オ4':0.6,'オ6':0.08,'オ7':0.62,'オ8':0.06,'オ9':0.28,'オ10':0.38,'オ12':0.46,'オ13':0.15,'オ14':0.71,'オ15':0.05,'オ16':0.37,'オ17':0.5,'オ18':0.05,'オ19':0.19,'オ20':0.05,
 'ワ1':0.65,'ワ5':0.35,'ワ6':0.25,'ワ7':0.19,'ワ8':0.74,'ワ9':0.16,'ワ10':0.3,'ワ11':0.41,'ワ12':0.04,'ワ13':0.06,'ワ14':0.06,'ワ15':0.27,'ワ16':0.09,'ワ17':1.06,'ワ18':0.02,
 'カ1':0.9,'カ2':1.45,'カ3':0.5,'カ4':0.62,'カ5':0.78}
PX2HA = (500/795)**2/1e4

# 1) 図面格子線(長い直線)をHoughで検出して消去
gl = cv2.HoughLinesP((im<140).astype(np.uint8)*255, 1, np.pi/720, threshold=600, minLineLength=900, maxLineGap=25)
grid_lines = []
dk = (im < 140)
for l in ([] if gl is None else gl):
    x1,y1,x2,y2 = [int(v) for v in np.asarray(l).ravel()]
    dx, dy = x2-x1, y2-y1
    if not (abs(dx) < 25 or abs(dy) < 35): continue      # 水平/垂直に近いものだけ
    # 太い線(林班界など)は除外: 法線方向±5pxが暗ければ太線
    L = int(np.hypot(dx,dy)); nx, ny = -dy/L, dx/L
    ts = np.linspace(0,1,200)
    px = (x1+dx*ts).astype(int); py = (y1+dy*ts).astype(int)
    thick = 0
    for off in (5,-5):
        qx = np.clip((px+nx*off).astype(int),0,W-1); qy = np.clip((py+ny*off).astype(int),0,H-1)
        thick = max(thick, dk[qy,qx].mean())
    if thick > 0.35: continue
    grid_lines.append((x1,y1,x2,y2))
# 格子線は「交差する線が無い所だけ」消す(交差線を切らない)
for (x1,y1,x2,y2) in grid_lines:
    dx, dy = x2-x1, y2-y1; L = int(np.hypot(dx,dy)); nx, ny = -dy/L, dx/L
    for t in np.linspace(0,1,L):
        px, py = x1+dx*t, y1+dy*t
        cross = False
        q = lambda off: dk[min(H-1,max(0,int(round(py+ny*off)))), min(W-1,max(0,int(round(px+nx*off))))]
        if q(3) and q(-3): cross = True   # 太線上は消さない
        for off in (4,5,6,-4,-5,-6):
            qx = int(round(px+nx*off)); qy = int(round(py+ny*off))
            if 0<=qx<W and 0<=qy<H and dk[qy,qx]: cross = True; break
        if not cross:
            cv2.circle(dark, (int(round(px)),int(round(py))), 2, 0, -1)
# 2) 濃い連続成分を残し、ラベル枠内の成分(数字)を除去
n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
w = stats[:,2]; h = stats[:,3]; bx0 = stats[:,0]; by0 = stats[:,1]; bx1 = bx0+w; by1 = by0+h
keep = np.maximum(w,h) >= 20; keep[0] = False
for sub,no,x0,y0,x1,y1,side in LOTS:
    if (sub,no) in (('オ',18),('ル',22)): continue
    keep[(bx0 >= x0+2) & (by0 >= y0+2) & (bx1 <= x1-2) & (by1 <= y1-2) & (np.minimum(w,h) >= 5)] = False
cand = keep[lab].astype(np.uint8)
# 2b) 太線(林班界・道路)は無条件に区画線
inv = (255-im).astype(np.float32)
thick_m = (cv2.GaussianBlur(inv,(0,0),1.0) > 150).astype(np.uint8)
n2, lab2, st2, _ = cv2.connectedComponentsWithStats(thick_m, connectivity=8)
k2 = np.maximum(st2[:,2], st2[:,3]) >= 80; k2[0] = False
thick_lines = k2[lab2].astype(np.uint8)
# 2c) 直線構造要素によるオープニング: 長さSLの直線が線の内側に収まる画素だけ残す(曲線＝等高線は残らない)
SL = CFG.get('straight_len', 31)
base = cv2.dilate(cand, np.ones((3,3),np.uint8))   # 線をやや太らせて途切れを防ぐ
straight = np.zeros((H,W), np.uint8)
for ang in np.arange(0, 180, 5):
    k = np.zeros((SL,SL), np.uint8)
    c = SL//2; dx = np.cos(np.radians(ang)); dy = np.sin(np.radians(ang))
    cv2.line(k, (int(round(c-dx*c)), int(round(c-dy*c))), (int(round(c+dx*c)), int(round(c+dy*c))), 1, 1)
    straight = np.maximum(straight, cv2.morphologyEx(base, cv2.MORPH_OPEN, k))
straight = cv2.erode(straight, np.ones((3,3),np.uint8)) | (straight & cand)
print('straight px', int(straight.sum()))
# 直線区間を検出して両端を延長し、角(頂点)で途切れた区画線を再接続する
EXT = CFG.get('extend', 14)
hs = cv2.HoughLinesP(straight*255, 1, np.pi/360, threshold=25, minLineLength=35, maxLineGap=6)
ext_img = np.zeros((H,W), np.uint8); nh = 0
for l in ([] if hs is None else hs):
    x1,y1,x2,y2 = [float(v) for v in np.asarray(l).ravel()]
    L = np.hypot(x2-x1, y2-y1)
    if L < 1: continue
    ux, uy = (x2-x1)/L, (y2-y1)/L
    p = (int(round(x1-ux*EXT)), int(round(y1-uy*EXT))); q = (int(round(x2+ux*EXT)), int(round(y2+uy*EXT)))
    cv2.line(ext_img, p, q, 1, 3); nh += 1
straight = np.maximum(straight, ext_img)
print('hough segs', nh)
ERASERS_A = CFG.get('erasers_A', [])
def finish(lines):
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(DI,DI)))
    M=24; lines[:M,:]=1; lines[-M:,:]=1; lines[:,:M]=1; lines[:,-M:]=1
    for poly in BARRIERS: cv2.polylines(lines, [np.array(poly, np.int32)], False, 1, thickness=DI)
    return lines
linesA = np.maximum(cand, thick_lines)
for (ex0,ey0,ex1,ey1) in ERASERS_A: linesA[ey0:ey1, ex0:ex1] = 0
linesA = finish(linesA)
linesB = np.maximum(thick_lines, straight)
for sub,no,x0,y0,x1,y1,side in LOTS:
    if (sub,no) in (('オ',18),('ル',22)): linesB[y0:y1, x0:x1] = np.maximum(linesB[y0:y1, x0:x1], cand[y0:y1, x0:x1])
linesB = finish(linesB)
lines = linesB
# 3) 自由空間の連結成分(A/B)
def components(lines):
    free = (lines==0).astype(np.uint8)
    nf, flab, fstats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    return free, flab, fstats[:,cv2.CC_STAT_AREA]
FREE = {'A': components(linesA), 'B': components(linesB)}
PREFER = CFG.get('prefer', {})   # 'A'/'B' を指定して固定
regions = {}; masks = {}
def extract(key, x0,y0,x1,y1, which):
    free, flab, farea = FREE[which]; exp_px = HA[key]/PX2HA
    if key in SEEDS:
        sx,sy = SEEDS[key]; comp = flab[sy,sx]; votes={comp:99}
    else:
        cx, cy = (x0+x1)/2, (y0+y1)/2; rx, ry = (x1-x0)/2, (y1-y0)/2
        votes = {}
        for extra, wt in ((-2,6),(0,5),(2,4),(4,3),(8,2),(12,1)):
            for a in range(0,360,10):
                x = int(cx + (rx+extra)*np.cos(np.radians(a))); y = int(cy + (ry+extra)*np.sin(np.radians(a)))
                if 0<=x<W and 0<=y<H and free[y,x]:
                    c = flab[y,x]; votes[c] = votes.get(c,0)+wt
        if not votes: return None
        cand_ = [c for c in votes if farea[c] <= 30*exp_px]
        if not cand_: cand_ = list(votes)
        comp = max(cand_, key=lambda c: (votes[c], -farea[c]))
    reg = flab==comp; area=int(farea[comp]); ha = area*PX2HA
    return dict(reg=reg, ha=ha, votes=int(votes[comp]), nvotes=int(sum(votes.values())))
for sub,no,x0,y0,x1,y1,side in LOTS:
    key=f'{sub}{no}'
    if key in MANUAL:
        reg = np.zeros((H,W), np.uint8)
        if 'ellipse' in MANUAL[key]:
            cx_,cy_,ax_,ay_ = MANUAL[key]['ellipse']; cv2.ellipse(reg, (cx_,cy_), (ax_,ay_), 0, 0, 360, 1, -1)
        else:
            cv2.fillPoly(reg, [np.array(MANUAL[key]['polygon'], np.int32)], 1)
        reg = reg.astype(bool); area=int(reg.sum()); ys,xs=np.where(reg); ha=area*PX2HA
        regions[key]=dict(src='M', ha=round(ha,3), reg_ha=HA[key], ok=True, manual=True, bbox=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())])
        masks[key]=reg; print(f'{key:5s} {ha:6.2f}ha / {HA[key]:5.2f}ha (手動)'); continue
    res = {w: extract(key,x0,y0,x1,y1,w) for w in ('A','B')}
    def score(w):
        r = res[w]
        if r is None: return 1e9
        return abs(np.log((r['ha']/HA[key])/0.85)) + (0.0 if w=='B' else 0.15)
    which = PREFER.get(key) or min(('A','B'), key=score)
    r = res[which]; reg = r['reg']; ha = r['ha']; ys,xs=np.where(reg)
    lo = 0.25 if HA[key] < 0.1 else 0.5
    ok = lo <= ha/HA[key] <= 2.0
    other = res['A' if which=='B' else 'B']
    regions[key]=dict(src=which, ha=round(ha,3), reg_ha=HA[key], ok=bool(ok), votes=r['votes'], nvotes=r['nvotes'],
                      other_ha=None if other is None else round(other['ha'],3),
                      bbox=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())])
    masks[key]=reg
    print(f'{key:5s} [{which}] {ha:6.2f}ha / {HA[key]:5.2f}ha 比={ha/HA[key]:6.2f}  (他方 {regions[key]["other_ha"]}) bbox={regions[key]["bbox"]}{"" if ok else "  <-- NG"}')
json.dump(regions, open('regions.json','w'), ensure_ascii=False)
np.savez_compressed('masks.npz', **{k: np.packbits(v) for k,v in masks.items()})
np.save('lines.npy', np.packbits(linesB)); np.save('linesA.npy', np.packbits(linesA))
print('NG:', sum(1 for r in regions.values() if not r['ok']))

def viz(name, crop, z):
    x0,y0,x1,y1 = crop
    sub = im[y0:y1, x0:x1]
    base = np.stack([np.clip(sub.astype(int)*0.5+128,0,255)]*3, -1).astype(np.uint8)
    l = lines[y0:y1, x0:x1].astype(bool); base[l] = (0,0,0)
    random.seed(1)
    for key,reg in masks.items():
        r = reg[y0:y1, x0:x1]
        if not r.any(): continue
        col = np.array([random.randint(60,255) for _ in range(3)])
        if not regions[key]['ok']: col = np.array([255,0,0])
        base[r] = (base[r]*0.4 + col*0.6).astype(np.uint8)
    pil = Image.fromarray(base).resize(((x1-x0)*z, (y1-y0)*z), Image.NEAREST)
    d = ImageDraw.Draw(pil); font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
    for x in range((x0//50)*50, x1+1, 50):
        if x<x0: continue
        X=(x-x0)*z; d.line([(X,0),(X,pil.height)], fill=(0,120,255), width=1); d.text((X+2,2), str(x), fill=(0,0,255), font=font)
    for y in range((y0//50)*50, y1+1, 50):
        if y<y0: continue
        Y=(y-y0)*z; d.line([(0,Y),(pil.width,Y)], fill=(0,120,255), width=1); d.text((2,Y+2), str(y), fill=(0,0,255), font=font)
    for sub_,no,lx0,ly0,lx1,ly1,side in LOTS:
        key=f'{sub_}{no}'; sx,sy=(lx0+lx1)//2,(ly0+ly1)//2
        if x0<=sx<x1 and y0<=sy<y1:
            d.text(((sx-x0)*z+4,(sy-y0)*z-16), key, fill=(200,0,0), font=font)
    pil.save(name)
for spec in sys.argv[1:]:
    name,c = spec.split('='); c=list(map(int,c.split(','))); viz(name+'.png', c[:4], c[4])
