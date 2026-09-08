# -*- coding: utf-8 -*-
"""森林計画図(スキャン画像)の位置合わせと、区画ポリゴンのKML/GeoJSON/HTML出力。
根拠: 図面の500m格子 = 旧日本測地系(Tokyo Datum) 平面直角座標VI系(EPSG:30166)。格子から縮尺・回転を、
基準点(惣社水分神社)から平行移動を求め、平行移動は格子(500m倍数)にスナップして確定する。
使い方: python3 georef_export.py 出力ディレクトリ   (カレントに masks.npz, regions.json, map_full.png が必要)"""
import sys, os, json, math, numpy as np, cv2
from pyproj import Transformer
from PIL import Image
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'annotate.py')).read()
ns = {}; exec(src[src.index('LOTS = ['):src.index('NOT_FOUND')], ns); LOTS = ns['LOTS']
HA = {'ル9':0.06,'ル15':0.22,'ル16':4.3,'ル17':0.09,'ル19':0.05,'ル20':0.77,'ル21':0.72,'ル22':0.06,
 'オ1':1.2,'オ3':0.04,'オ4':0.6,'オ6':0.08,'オ7':0.62,'オ8':0.06,'オ9':0.28,'オ10':0.38,'オ12':0.46,'オ13':0.15,'オ14':0.71,'オ15':0.05,'オ16':0.37,'オ17':0.5,'オ18':0.05,'オ19':0.19,'オ20':0.05,
 'ワ1':0.65,'ワ5':0.35,'ワ6':0.25,'ワ7':0.19,'ワ8':0.74,'ワ9':0.16,'ワ10':0.3,'ワ11':0.41,'ワ12':0.04,'ワ13':0.06,'ワ14':0.06,'ワ15':0.27,'ワ16':0.09,'ワ17':1.06,'ワ18':0.02,
 'カ1':0.9,'カ2':1.45,'カ3':0.5,'カ4':0.62,'カ5':0.78}

# ---- 1) 図面の格子線(画像座標) → 局所格子座標のアフィン ----
Hsegs = [((803,794),(3275,773)),((1653,789),(3139,776)),((19,802),(1290,791)),((535,799),(1584,790)),
         ((886,1584),(3279,1563)),((24,1593),(1530,1580))]
Hidx  = [0,0,0,0,1,1]
Vsegs = [((1092,465),(1102,1662)),((1877,24),(1886,1096)),((1882,500),(1894,1861)),((2668,89),(2676,993)),((2673,925),(2687,2017))]
Vidx  = [0,1,1,2,2]
def fit_line(pts):
    P = np.array(pts, float); c = P.mean(0); u,s,vt = np.linalg.svd(P-c); return c, vt[0]
def intersect(l1, l2):
    c1,d1 = l1; c2,d2 = l2; t = np.linalg.solve(np.array([d1,-d2]).T, c2-c1); return c1 + d1*t[0]
lh = {j: fit_line([p for s,k in zip(Hsegs,Hidx) if k==j for p in s]) for j in set(Hidx)}
lv = {i: fit_line([p for s,k in zip(Vsegs,Vidx) if k==i for p in s]) for i in set(Vidx)}
px, loc = [], []
for i in lv:
    for j in lh:
        px.append(intersect(lv[i], lh[j])); loc.append((500*i, -500*j))
px = np.array(px); loc = np.array(loc, float)
M = np.hstack([px, np.ones((len(px),1))]); coef, *_ = np.linalg.lstsq(M, loc, rcond=None); A = coef.T
# ---- 2) 基準点で平行移動を求め、500m格子にスナップ ----
EPSG_GRID = 30166   # Tokyo / Japan Plane Rectangular CS VI
GCP_IMG = (1865, 2255)                       # 図面上の惣社水分神社(鳥居記号)
GCP_LONLAT = (136.018349, 34.462065)         # 地理院地図上の同神社 (WGS84)
to_grid = Transformer.from_crs('EPSG:4326', f'EPSG:{EPSG_GRID}', always_xy=True)
to_geo  = Transformer.from_crs(f'EPSG:{EPSG_GRID}', 'EPSG:4326', always_xy=True)
E, N = to_grid.transform(*GCP_LONLAT)
off = np.array([E, N]) - A @ np.array([GCP_IMG[0], GCP_IMG[1], 1.0])
off_snap = np.round(off/500)*500
print('scale m/px', round(float(np.hypot(A[0,0],A[1,0])),4), 'rot deg', round(math.degrees(math.atan2(A[1,0],A[0,0])),3))
print('GCP offset', off.round(1), '-> snapped', off_snap, ' residual(m)', (off-off_snap).round(1))
def img_to_lonlat(x, y):
    p = A @ np.array([x, y, 1.0]) + off_snap
    return to_geo.transform(p[0], p[1])
json.dump({'A': A.tolist(), 'offset': off_snap.tolist(), 'grid_epsg': EPSG_GRID, 'gcp_img': GCP_IMG, 'gcp_lonlat': GCP_LONLAT,
           'snap_residual_m': (off-off_snap).tolist()}, open(os.path.join(OUT,'georef.json'),'w'), ensure_ascii=False, indent=1)

# ---- 3) 区画ポリゴン(render_regions.py と同じ整形) ----
regions = json.load(open('regions.json')); mz = np.load('masks.npz')
im = Image.open('map_full.png'); W, H = im.size
raw = {f'{s}{n}': np.unpackbits(mz[f'{s}{n}'])[:H*W].reshape(H,W).astype(bool) for s,n,*_ in LOTS}
order = sorted(raw, key=lambda k: raw[k].sum())
for i, small in enumerate(order):
    for big in order[i+1:]:
        if (raw[small] & raw[big]).any(): raw[big] &= ~raw[small]
feats = []
for sub,no,*_ in LOTS:
    key = f'{sub}{no}'
    m = raw[key].astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    cs,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = [cv2.approxPolyDP(c, 2.0, True).reshape(-1,2) for c in cs if cv2.contourArea(c) > 150]
    r = regions[key]
    note = '手動推定' if r.get('manual') else ('概略' if not r.get('ok') else '')
    if key == 'ワ1': note = '図面上端で切れているため北側不完全'
    polys = [[img_to_lonlat(float(x), float(y)) for x,y in c] for c in cs]
    feats.append(dict(key=key, sub=sub, no=no, ha=HA[key], note=note, polys=polys))

# ---- 4) 出力: GeoJSON / KML(区画) / KMZ(図面画像の重ね) / HTML ----
gj = {'type':'FeatureCollection','features':[]}
for f in feats:
    for poly in f['polys']:
        ring = [[round(lo,7), round(la,7)] for lo,la in poly]; ring.append(ring[0])
        gj['features'].append({'type':'Feature','properties':{'name':f['key'],'小班':f['sub'],'施業番号':f['no'],'森林簿面積ha':f['ha'],'備考':f['note']},
                               'geometry':{'type':'Polygon','coordinates':[ring]}})
json.dump(gj, open(os.path.join(OUT,'林班57_区画.geojson'),'w'), ensure_ascii=False)

def kml_poly(f):
    parts = []
    for poly in f['polys']:
        coords = ' '.join(f'{lo:.7f},{la:.7f},0' for lo,la in poly) + f' {poly[0][0]:.7f},{poly[0][1]:.7f},0'
        parts.append(f'<Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}</coordinates></LinearRing></outerBoundaryIs></Polygon>')
    geom = parts[0] if len(parts)==1 else '<MultiGeometry>' + ''.join(parts) + '</MultiGeometry>'
    desc = f'林班57 小班{f["sub"]} 施業番号{f["no"]}　森林簿面積 {f["ha"]} ha' + (f'　※{f["note"]}' if f['note'] else '')
    return f'<Placemark><name>{f["key"]}</name><description>{desc}</description><styleUrl>#lot</styleUrl>{geom}</Placemark>'
kml = ['<?xml version="1.0" encoding="UTF-8"?>','<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>林班57 森林簿記載区画（宇陀市菟田野上芳野）</name>',
       '<Style id="lot"><LineStyle><color>ff0000dd</color><width>3</width></LineStyle><PolyStyle><color>400000ff</color></PolyStyle></Style>']
for sub in ['ル','オ','ワ','カ']:
    kml.append(f'<Folder><name>小班{sub}</name>')
    kml += [kml_poly(f) for f in feats if f['sub']==sub]
    kml.append('</Folder>')
kml.append('</Document></kml>')
open(os.path.join(OUT,'林班57_区画.kml'),'w').write('\n'.join(kml))

# 図面画像の重ね合わせ(GroundOverlay, Google Earth用)。回転付きLatLonBoxで近似
import zipfile
cx, cy = W/2, H/2
clon, clat = img_to_lonlat(cx, cy)
sx = float(np.hypot(A[0,0],A[1,0])); rot = math.degrees(math.atan2(A[1,0],A[0,0]))   # 画像x軸の東からの回転(北向き正の座標系)
mlat = 111132.954 - 559.822*math.cos(2*math.radians(clat)); mlon = 111412.84*math.cos(math.radians(clat)) - 93.5*math.cos(3*math.radians(clat))
half_w_deg = (W*sx/2)/mlon; half_h_deg = (H*sx/2)/mlat
img = Image.open('annotated_p1.png') if os.path.exists('annotated_p1.png') else im
img = img.crop((0,0,W,H)).convert('RGB'); img.save(os.path.join(OUT,'map_overlay.jpg'), quality=80)
gkml = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>森林計画図（赤枠付き）重ね合わせ</name>
<GroundOverlay><name>森林計画図</name><color>b0ffffff</color><Icon><href>map_overlay.jpg</href></Icon>
<LatLonBox><north>{clat+half_h_deg:.7f}</north><south>{clat-half_h_deg:.7f}</south><east>{clon+half_w_deg:.7f}</east><west>{clon-half_w_deg:.7f}</west><rotation>{-rot:.4f}</rotation></LatLonBox>
</GroundOverlay></Document></kml>'''
with zipfile.ZipFile(os.path.join(OUT,'森林計画図_重ね合わせ.kmz'),'w',zipfile.ZIP_DEFLATED) as z:
    z.writestr('doc.kml', gkml); z.write(os.path.join(OUT,'map_overlay.jpg'), 'map_overlay.jpg')
os.remove(os.path.join(OUT,'map_overlay.jpg'))

# HTML(Leaflet + 地理院タイル/OSM)
html = f'''<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8"><title>林班57 区画位置図</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>html,body,#map{{height:100%;margin:0}} .lbl{{background:rgba(255,255,255,.85);border:1px solid #d00;color:#d00;font-weight:bold;font-size:12px;padding:1px 3px;white-space:nowrap}}</style></head>
<body><div id="map"></div><script>
var gj = {json.dumps(gj, ensure_ascii=False)};
var map = L.map('map');
var gsi = L.tileLayer('https://cyberjapandata.gsi.go.jp/xyz/std/{{z}}/{{x}}/{{y}}.png', {{maxZoom:18, attribution:'<a href="https://maps.gsi.go.jp/development/ichiran.html">地理院タイル</a>'}}).addTo(map);
var photo = L.tileLayer('https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{{z}}/{{x}}/{{y}}.jpg', {{maxZoom:18, attribution:'地理院タイル(写真)'}});
var osm = L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{maxZoom:19, attribution:'&copy; OpenStreetMap contributors'}});
var layer = L.geoJSON(gj, {{style:{{color:'#dd0000',weight:2,fillColor:'#ff4444',fillOpacity:0.25}},
  onEachFeature:function(f,l){{ var p=f.properties; l.bindPopup('<b>'+p.name+'</b><br>小班'+p['小班']+' 施業番号'+p['施業番号']+'<br>森林簿面積 '+p['森林簿面積ha']+' ha'+(p['備考']?'<br>※'+p['備考']:''));
    l.bindTooltip(p.name,{{permanent:true,direction:'center',className:'lbl'}}); }} }}).addTo(map);
L.control.layers({{'地理院地図':gsi,'航空写真':photo,'OpenStreetMap':osm}},{{'区画':layer}}).addTo(map);
map.fitBounds(layer.getBounds());
</script></body></html>'''
open(os.path.join(OUT,'林班57_区画位置図.html'),'w').write(html)
print('features', len(feats), 'written to', OUT)
