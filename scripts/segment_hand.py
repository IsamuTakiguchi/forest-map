import numpy as np, cv2, json, sys, random
from PIL import Image, ImageDraw, ImageFont
im = np.array(Image.open('map_full.png')); H,W = im.shape
red = np.unpackbits(np.load('hand_red_map.npy'))[:H*W].reshape(H,W).astype(np.uint8)
src = open('/home/user/forest-map/scripts/annotate.py').read()
ns = {}; exec(src[src.index('LOTS = ['):src.index('NOT_FOUND')], ns); LOTS = ns['LOTS']
HA = {'ル9':0.06,'ル15':0.22,'ル16':4.3,'ル17':0.09,'ル19':0.05,'ル20':0.77,'ル21':0.72,'ル22':0.06,
 'オ1':1.2,'オ3':0.04,'オ4':0.6,'オ6':0.08,'オ7':0.62,'オ8':0.06,'オ9':0.28,'オ10':0.38,'オ12':0.46,'オ13':0.15,'オ14':0.71,'オ15':0.05,'オ16':0.37,'オ17':0.5,'オ18':0.05,'オ19':0.19,'オ20':0.05,
 'ワ1':0.65,'ワ5':0.35,'ワ6':0.25,'ワ7':0.19,'ワ8':0.74,'ワ9':0.16,'ワ10':0.3,'ワ11':0.41,'ワ12':0.04,'ワ13':0.06,'ワ14':0.06,'ワ15':0.27,'ワ16':0.09,'ワ17':1.06,'ワ18':0.02,
 'カ1':0.9,'カ2':1.45,'カ3':0.5,'カ4':0.62,'カ5':0.78}
PX2HA = (500/795)**2/1e4
CFG = json.load(open('hand_config.json')) if __import__('os').path.exists('hand_config.json') else {}
RD, ND = CFG.get('red_dilate', 7), CFG.get('near', 25)
# 黒線(数字除去)
dark = (im < 90).astype(np.uint8)
n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
w = stats[:,2]; h = stats[:,3]; bx0 = stats[:,0]; by0 = stats[:,1]; bx1 = bx0+w; by1 = by0+h
keep = np.maximum(w,h) >= 20; keep[0] = False
for sub,no,x0,y0,x1,y1,side in LOTS:
    if (sub,no) in (('オ',18),('ル',22)): continue
    keep[(bx0 >= x0+2) & (by0 >= y0+2) & (bx1 <= x1-2) & (by1 <= y1-2) & (np.minimum(w,h) >= 5)] = False
black = keep[lab].astype(np.uint8)
k = lambda d: cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(d,d))
from skimage.morphology import skeletonize
red_sk = skeletonize(red.astype(bool))
black_sk = skeletonize(black.astype(bool))
sel = black_sk & cv2.dilate(red, k(ND)).astype(bool) if ND > 0 else np.zeros_like(red_sk)
# 太線(林道・林班界)も境界に加える
inv = (255-im).astype(np.float32)
thick_m = (cv2.GaussianBlur(inv,(0,0),1.0) > 150).astype(np.uint8)
n2, lab2, st2, _ = cv2.connectedComponentsWithStats(thick_m, connectivity=8)
k2 = np.maximum(st2[:,2], st2[:,3]) >= 80; k2[0] = False
thick_sk = skeletonize(k2[lab2]) if CFG.get('use_thick', True) else np.zeros_like(red_sk)
core = (red_sk | sel | thick_sk).astype(np.uint8)
core = cv2.dilate(core, k(3))
# 途切れの橋渡し: 骨格の端点から進行方向に最大GAP px 伸ばし、他の線に当たれば接続
GAP = CFG.get('gap', 45)
sk = skeletonize(core.astype(bool)).astype(np.uint8)
nb = cv2.filter2D(sk, -1, np.ones((3,3),np.float32)) - sk
ends = np.argwhere((sk==1) & (nb==1))
bridges = np.zeros_like(core); nbr = 0
for (y,x) in ends:
    # 端点から骨格を10px辿って方向を推定
    path=[(y,x)]; seen={(y,x)}; cur=(y,x)
    for _ in range(12):
        nxt=[(cur[0]+dy,cur[1]+dx) for dy in (-1,0,1) for dx in (-1,0,1) if (dy or dx) and 0<=cur[0]+dy<H and 0<=cur[1]+dx<W and sk[cur[0]+dy,cur[1]+dx] and (cur[0]+dy,cur[1]+dx) not in seen]
        if not nxt: break
        cur=nxt[0]; seen.add(cur); path.append(cur)
    if len(path) < 6: continue
    dy, dx = y-path[-1][0], x-path[-1][1]; L=np.hypot(dy,dx)
    if L < 3: continue
    dy/=L; dx/=L
    hit=None
    for t in range(6, GAP):
        for off in (-0.35,0,0.35):
            a = np.arctan2(dy,dx)+off
            py_, px_ = int(round(y+np.sin(a)*t)), int(round(x+np.cos(a)*t))
            if 0<=py_<H and 0<=px_<W and core[py_,px_]: hit=(py_,px_); break
        if hit: break
    if hit: cv2.line(bridges, (x,y), (hit[1],hit[0]), 1, 3); nbr+=1
print('endpoints', len(ends), 'bridges', nbr)
barrier = cv2.dilate(np.maximum(core, bridges), k(RD))
for poly in CFG.get('barriers', []): cv2.polylines(barrier, [np.array(poly, np.int32)], False, 1, 5)
for (ex0,ey0,ex1,ey1) in CFG.get('erasers', []): barrier[ey0:ey1, ex0:ex1] = 0
M=24; barrier[:M,:]=1; barrier[-M:,:]=1; barrier[:,:M]=1; barrier[:,-M:]=1
free = (barrier==0).astype(np.uint8)
nf, flab, fstats, _ = cv2.connectedComponentsWithStats(free, connectivity=4); farea = fstats[:,4]
barrier_r = cv2.dilate(red_sk.astype(np.uint8), k(RD)); barrier_r[:24,:]=1; barrier_r[-24:,:]=1; barrier_r[:,:24]=1; barrier_r[:,-24:]=1
free_r = (barrier_r==0).astype(np.uint8)
nfr, flab_r, fst_r, _ = cv2.connectedComponentsWithStats(free_r, connectivity=4); farea_r = fst_r[:,4]
RED_ONLY = set(CFG.get('red_only', []))
regions, masks = {}, {}
SEEDS = CFG.get('seeds', {}); MANUAL = CFG.get('manual', {})
for sub,no,x0,y0,x1,y1,side in LOTS:
    key=f'{sub}{no}'
    if key in MANUAL:
        reg = np.zeros((H,W), np.uint8)
        if isinstance(MANUAL[key], dict): cx_,cy_,ax_,ay_ = MANUAL[key]['ellipse']; cv2.ellipse(reg, (cx_,cy_), (ax_,ay_), 0, 0, 360, 1, -1)
        else: cv2.fillPoly(reg, [np.array(MANUAL[key], np.int32)], 1)
        reg = reg.astype(bool)
        ha = reg.sum()*PX2HA; regions[key] = dict(ha=round(ha,3), reg_ha=HA[key], ok=True, manual=True); masks[key]=reg
        print(f'{key:5s} {ha:6.2f}/{HA[key]:5.2f} (手動)'); continue
    fl, fa, fr = (flab_r, farea_r, free_r) if key in RED_ONLY else (flab, farea, free)
    if key in SEEDS:
        sx,sy = SEEDS[key]; comp = fl[sy,sx]
    else:
        cx, cy = (x0+x1)/2, (y0+y1)/2; rx, ry = (x1-x0)/2, (y1-y0)/2; votes = {}
        for extra, wt in ((-2,6),(0,5),(2,4),(4,3),(8,2),(12,1)):
            for a in range(0,360,10):
                x = int(cx + (rx+extra)*np.cos(np.radians(a))); y = int(cy + (ry+extra)*np.sin(np.radians(a)))
                if 0<=x<W and 0<=y<H and fr[y,x]: c = fl[y,x]; votes[c] = votes.get(c,0)+wt
        exp = HA[key]/PX2HA
        cand = [c for c in votes if fa[c] <= 30*exp] or list(votes)
        comp = max(cand, key=lambda c: (votes[c], -fa[c]))
    reg = fl==comp; ha = fa[comp]*PX2HA
    lo = 0.25 if HA[key] < 0.1 else 0.5; ok = lo <= ha/HA[key] <= 2.0
    ys,xs = np.where(reg)
    regions[key] = dict(ha=round(float(ha),3), reg_ha=HA[key], ok=bool(ok), bbox=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())]); masks[key]=reg
    print(f'{key:5s} {ha:6.2f}/{HA[key]:5.2f} 比={ha/HA[key]:5.2f} bbox={regions[key]["bbox"]}{"" if ok else "  <-- NG"}')
print('NG', sum(1 for r in regions.values() if not r['ok']))
json.dump(regions, open('regions.json','w'), ensure_ascii=False)
np.savez_compressed('masks.npz', **{k: np.packbits(v) for k,v in masks.items()})
def viz(name, crop, z):
    x0,y0,x1,y1 = crop; sub = im[y0:y1,x0:x1]
    base = np.stack([np.clip(sub.astype(int)*0.5+128,0,255)]*3,-1).astype(np.uint8)
    base[barrier[y0:y1,x0:x1].astype(bool)] = (0,0,0); random.seed(1)
    for key,reg in masks.items():
        r = reg[y0:y1,x0:x1]
        if not r.any(): continue
        col = np.array([255,0,0]) if not regions[key]['ok'] else np.array([random.randint(60,255) for _ in range(3)])
        base[r] = (base[r]*0.4+col*0.6).astype(np.uint8)
    pil = Image.fromarray(base).resize(((x1-x0)*z,(y1-y0)*z), Image.NEAREST); d = ImageDraw.Draw(pil)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
    for x in range((x0//50)*50, x1+1, 50):
        if x>=x0: X=(x-x0)*z; d.line([(X,0),(X,pil.height)], fill=(0,120,255)); d.text((X+2,2), str(x), fill=(0,0,255), font=font)
    for y in range((y0//50)*50, y1+1, 50):
        if y>=y0: Y=(y-y0)*z; d.line([(0,Y),(pil.width,Y)], fill=(0,120,255)); d.text((2,Y+2), str(y), fill=(0,0,255), font=font)
    for s_,n_,lx0,ly0,lx1,ly1,_ in LOTS:
        sx,sy=(lx0+lx1)//2,(ly0+ly1)//2
        if x0<=sx<x1 and y0<=sy<y1: d.text(((sx-x0)*z+4,(sy-y0)*z-16), f'{s_}{n_}', fill=(200,0,0), font=font)
    pil.save(name)
for spec in sys.argv[1:]:
    name,c = spec.split('='); c=list(map(int,c.split(','))); viz(name+'.png', c[:4], c[4])
