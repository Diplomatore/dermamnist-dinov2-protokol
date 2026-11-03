#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""defter_uret_v1.py — ARŞİV (DONDURULMUŞ, 2026-10-11 ~12:10): v1 tek defter 3-1-colab.ipynb'nin üreticisi.
kc1-tekrar oturumu bu defterle koşuyor (SHA256 fc6a3f28…; Drive kopyası Colab otomatik kaydıyla çıktı içerir).
v2 (paralel sekmeler 3-1-colab-A/B/C.ipynb) → defter_uret.py. Bu dosya yalnız kaynak kaydıdır; drive_paketle v1 defteri
Drive'a YAZMAZ. Kullanım (yalnız yeniden üretim denetimi): <uv-python> colab/hat/defter_uret_v1.py"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HAT = Path(__file__).resolve().parent

MD0 = r"""# 3.1 DermaMNIST — Colab hattı (tek defter)

**Ne yapar:** Drive'daki kod paketini (`yz/3-1/`) doğrular, veriyi indirir, ortamı sınar ve ilk hücredeki **KUYRUK** grubunu
kademeli koşar: duman → ön-uçuş → tam. Aşama kapıları otomatiktir. Beklenti ihlalinde tam koşum **başlamaz**.

| KUYRUK | içerik | tahmini L4 süresi |
|---|---|---|
| `kc1-tekrar` | **Tekrar kapısı:** rakip (baseline), H01, H02 ve H08 yerel tarifleri, seed 0–4 (32 koşum) | ≈ 2,5–4 sa |
| `kc2-tur4` | Tur-4: H12 → H10 → H11 (3 kuyruk × 8 koşum). **Şef izni gerekir** (`kuyruk/kc2-tur4.izin.json`) | ≈ 1,5–3 sa |

**Senin yapacağın:** *Çalışma zamanı → Çalışma zamanı türünü değiştir → **L4 GPU*** → *Tümünü çalıştır* → sekmeyi açık tut.
Oturum koparsa defteri yeniden aç, yine *Tümünü çalıştır*. Devam mantığı bitmiş koşumları atlar, yalnız kalanları koşar.
Ön-kayıt değişmez: beklenti dosyası bayt bayt kopyalanır.

Her hücre hata verirse tam traceback Drive'a yazılır: `yz/3-1/durum/hata_<zaman>.log`. Kuyruk koşarken her 60 sn'de
`yz/3-1/durum/canli.json` güncellenir. Şef bunu kendi bilgisayarından izler.
"""

C1 = r"""# 1) AYARLAR + Drive + hata günlüğü — YALNIZ KUYRUK satırını değiştir (şef söyler)
KUYRUK = 'kc1-tekrar'      # 'kc1-tekrar' (tekrar kapısı) → kapı geçince 'kc2-tur4' (H12 → H10 → H11)
DURDA_DEVAM = False        # grup içinde bir kuyruk aşama kapısında DURURSA kalan kuyruklar koşmaz (şef teşhis eder)

import os, sys, json, time, shutil, subprocess, traceback, pathlib, datetime, hashlib
os.environ['TZ'] = 'Europe/Istanbul'; time.tzset()   # run_id ve damgalar yerel saatle (yerel koşumlarla kıyaslanır)
from google.colab import drive
drive.mount('/content/drive')
YZ = pathlib.Path('<DRIVE_KOK>')
DRIVE = YZ / '3-1'              # kalıcı çalışma kökü (kod paketi + runs/ kuyruk/ sonuclar/ env/ durum/)
IS = pathlib.Path('/content/is')    # oturumluk çalışma kopyası (kod Drive'dan kopyalanır, SHA256 doğrulanır)
VERI = pathlib.Path('/content/veri')  # oturumluk veri (Zenodo'dan indirilir, MD5 + SHA256)
DURUM = DRIVE / 'durum'
assert (DRIVE / 'PAKET.json').exists(), (f'{DRIVE}/PAKET.json yok. Drive eşitlemesi bitmemiş olabilir; '
                                        'birkaç dakika bekleyip defteri yeniden çalıştır.')
DURUM.mkdir(parents=True, exist_ok=True)
OTURUM = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

def _hata_kaydet(shell, etype, value, tb, tb_offset=None):
    # Her hücre hatası: tam traceback → yz/3-1/durum/hata_<zaman>.log (şefin yerel izleyicisi HATA-LOG olayı basar)
    satirlar = traceback.format_exception(etype, value, tb)
    try:
        hucre = (shell.user_ns.get('In') or [''])[-1]
        yol = DURUM / f'hata_{datetime.datetime.now():%Y%m%d-%H%M%S}.log'
        yol.write_text(f'oturum {OTURUM} · kuyruk {KUYRUK}\n--- hücre ---\n{hucre}\n--- traceback ---\n' + ''.join(satirlar),
                       encoding='utf-8')
        satirlar = satirlar + [f'\n[hata günlüğü] {yol}\n']
    except Exception as e:
        satirlar = satirlar + [f'\n[hata günlüğü YAZILAMADI: {e}]\n']
    return satirlar
get_ipython().set_custom_exc((Exception,), _hata_kaydet)

def kos(cmd, cwd=None, env=None):
    # Komutu koşar, çıktıyı canlı basar, dönüş kodunu verir
    cmd = [str(c) for c in cmd]
    print('$', ' '.join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
    for satir in p.stdout:
        print(satir, end='', flush=True)
    return p.wait()

print('oturum', OTURUM, '· kuyruk grubu', KUYRUK, '· Drive', DRIVE)
"""

C2 = r"""# 2) DONANIM: yalnız NVIDIA L4 (ön-kayıt: tüm Colab kolları aynı sınıfta). L4 değilse DUR.
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip()
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
assert 'L4' in gpu, (f'GPU L4 DEĞİL: {gpu!r}. Çalışma zamanı > Çalışma zamanı türünü değiştir > L4 GPU seç, '
                     'sonra defteri baştan çalıştır. Başka GPU ile ölçüm YAPILMAZ.')
mesgul = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,name', '--format=csv,noheader'],
                        capture_output=True, text=True).stdout.strip()
assert not mesgul, f'GPU başka bir süreçle meşgul: {mesgul}'
print('GPU OK:', gpu)
"""

C3 = r"""# 3) ORTAM: Colab'ın kendi torch'u (pin YOK; sürümler kaydedilir) + medmnist/timm/tensorboardX pinleri.
#    TF32 kapalı + deterministik ortam değişkenleri (tüm alt süreçler devralır). Defter çekirdeği torch'u İÇE AKTARMAZ.
SURUM_KOD = r'''
import json, platform, importlib
d = {'python': platform.python_version()}
for m in ('torch', 'torchvision', 'numpy', 'scipy', 'sklearn', 'pandas', 'PIL', 'timm', 'medmnist', 'tensorboardX', 'huggingface_hub'):
    try:
        d[m] = importlib.import_module(m).__version__
    except Exception as e:
        d[m] = None
import torch
d.update({'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'cuda_var': torch.cuda.is_available(),
          'cihaz': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
print(json.dumps(d))
'''
def surumler():
    r = subprocess.run(['python', '-c', SURUM_KOD], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])

once = surumler()
assert kos(['python', '-m', 'pip', 'install', '-q', 'medmnist==3.0.2', 'timm==1.0.30', 'tensorboardX==2.6.5']) == 0, 'pip başarısız'
ortam = surumler()
print(json.dumps(ortam, indent=1, ensure_ascii=False))
assert ortam['torch'] == once['torch'], f"pip torch'u değiştirdi: {once['torch']} → {ortam['torch']} (DUR)"
for k, v in {'medmnist': '3.0.2', 'timm': '1.0.30', 'tensorboardX': '2.6.5'}.items():
    assert str(ortam[k]) == v, f'{k} pinli sürüm değil: {ortam[k]} ≠ {v}'
assert ortam['cuda_var'] and 'L4' in str(ortam['cihaz']), f'torch CUDA/L4 görmüyor: {ortam}'

# Ortam değişkenleri: TF32 kütüphane düzeyinde kapalı (cuBLAS/cuDNN), cuBLAS deterministik çalışma alanı, UTF-8.
os.environ.update({'NVIDIA_TF32_OVERRIDE': '0', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONIOENCODING': 'utf-8',
                   'HF_HUB_DISABLE_TELEMETRY': '1'})
ortam.update({'zaman': datetime.datetime.now().isoformat(timespec='seconds'), 'oturum': OTURUM, 'kuyruk': KUYRUK,
              'nvidia_smi': gpu, 'ortam_degiskenleri': {k: os.environ[k] for k in ('NVIDIA_TF32_OVERRIDE', 'CUBLAS_WORKSPACE_CONFIG', 'TZ')}})
ENV = DRIVE / 'env'; ENV.mkdir(exist_ok=True)
(ENV / 'ortam-colab.json').write_text(json.dumps(ortam, indent=1, ensure_ascii=False), encoding='utf-8')
with open(ENV / 'ortam-colab-oturumlar.jsonl', 'a', encoding='utf-8') as f:
    f.write(json.dumps(ortam, ensure_ascii=False) + '\n')
(ENV / f'requirements-colab-{OTURUM}.lock').write_text(
    subprocess.run(['python', '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout, encoding='utf-8')
# Sürüm sürüklenmesi: ilk Colab oturumuna göre. Başlamış bir kuyruğun DEVAMINDA çekirdek sürüm değiştiyse DUR.
ILK = ENV / 'ortam-colab-ilk.json'
CEKIRDEK = ('python', 'torch', 'cuda', 'cudnn', 'torchvision', 'numpy', 'timm', 'medmnist', 'sklearn')
if not ILK.exists():
    ILK.write_text(json.dumps(ortam, indent=1, ensure_ascii=False), encoding='utf-8'); print('ilk Colab ortamı kaydedildi')
else:
    ilk = json.loads(ILK.read_text(encoding='utf-8'))
    fark = {k: (ilk.get(k), ortam.get(k)) for k in CEKIRDEK if ilk.get(k) != ortam.get(k)}
    if fark:
        print('!!! UYARI: Colab ortamı ilk oturumdan FARKLI:', fark)
        gruplar_ = json.loads((DRIVE / 'kuyruk' / 'gruplar.json').read_text(encoding='utf-8'))['gruplar']
        baslamis = [p.name for ad in gruplar_.get(KUYRUK, []) for p in (DRIVE / 'runs' / 'kuyruklar').glob(f'{ad}*.baslat.json')] \
            if (DRIVE / 'runs' / 'kuyruklar').exists() else []
        assert not baslamis, (f'Başlamış kuyruk {baslamis} farklı çekirdek sürümle sürdürülemez (seed\'ler iki ortama bölünür). '
                              f'DUR → şefe bildir. Fark: {fark}')
    else:
        print('ortam ilk Colab oturumuyla aynı (çekirdek sürümler)')
"""

C4 = r"""# 4) PAKET BÜTÜNLÜĞÜ: Drive'daki kod → /content/is (kopya), her dosya PAKET.json SHA256'sıyla doğrulanır.
#    runs/ kuyruk/ sonuclar/ env/ → Drive'a bağ (kalıcı); data/medmnist, data/dermamnist_c → /content/veri.
paket = json.loads((DRIVE / 'PAKET.json').read_text(encoding='utf-8'))
yardim_rel = 'colab/defter_yardim.py'
gecici = pathlib.Path('/tmp/defter_yardim_dogrula.py')
shutil.copyfile(DRIVE / yardim_rel, gecici)
assert hashlib.sha256(gecici.read_bytes()).hexdigest() == paket['dosyalar'][yardim_rel]['sha256'], \
    'defter_yardim.py SHA256 PAKET.json ile uyuşmuyor (Drive eşitlemesi yarım?) — DUR'
IS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, '/tmp'); import importlib; DY = importlib.import_module('defter_yardim_dogrula')
pk = DY.paket_kopyala_dogrula(DRIVE, IS)
print(json.dumps({k: v for k, v in pk.items() if k != 'fazla'}, ensure_ascii=False))
assert not pk['bozuk'] and not pk['eksik'], f"PAKET doğrulaması BAŞARISIZ: bozuk {pk['bozuk'][:5]} eksik {pk['eksik'][:5]}"
if pk['fazla']:
    print('UYARI: pakette olmayan dosyalar (kullanılmaz):', pk['fazla'][:10])
for rel, b in paket.get('kuyruk_dosyalari', {}).items():   # ön-kayıtlı kuyruk + beklenti dosyaları Drive'da bayt bayt aynı mı
    assert hashlib.sha256((DRIVE / rel).read_bytes()).hexdigest() == b['sha256'], \
        f'{rel} PAKET.json ile uyuşmuyor (ön-kayıt değişmiş ya da Drive eşitlemesi yarım) — DUR, şefe bildir'
print('kuyruk/beklenti dosyaları doğrulandı:', len(paket.get('kuyruk_dosyalari', {})))
bag = DY.baglantilari_kur(DRIVE, IS, VERI)
print('bağlar:', json.dumps(bag, ensure_ascii=False))
with open(DRIVE / 'env' / 'paket-oturumlar.jsonl', 'a', encoding='utf-8') as f:
    f.write(json.dumps({'oturum': OTURUM, 'kuyruk': KUYRUK, 'surum': pk['surum'], 'dosya': pk['kopyalanan'],
                        'zaman': datetime.datetime.now().isoformat(timespec='seconds')}, ensure_ascii=False) + '\n')
os.chdir(IS)
print('PAKET OK · sürüm', pk['surum'], '· çalışma kökü', os.getcwd())
"""

C5 = r"""# 5) VERİ: Zenodo → /content/veri (oturum başına), resmî MD5 + yerel SHA256 + bayt. 64/128 gerekmez.
#    DermaMNIST-C (28/224) arka planda iner (kuyruk beklemez; K5 içindir) → durum/veri_c.json.
vm = DY.veri_hazirla(IS / 'colab' / 'veri_listesi.json', VERI, 'medmnist', DURUM / f'veri_medmnist_{OTURUM}.json')
assert all(v['dogrulandi'] for v in vm.values()), vm
VERI_C_IPLIK = DY.veri_arka_plan(IS / 'colab' / 'veri_listesi.json', VERI, 'dermamnist_c', DURUM / 'veri_c.json',
                                 DURUM / f'veri_c_{OTURUM}.log')
print('MedMNIST 28/224 doğrulandı; DermaMNIST-C arka planda →', DURUM / 'veri_c.json')
"""

C6 = r"""# 6) ÖNAĞIRLIKLAR + SAYISAL ÖZ-SINAMA (ölçüm değil): DINOv2-S/B + ResNet-18 SHA256 = yerel; TF32 gerçekten kapalı mı
#    (fp64 referansına göre göreli hata); aynı seed iki kez → bit bit aynı mı (R18 ve ViT, küçük alt küme).
AG_KOD = r'''
import json, hashlib, torchvision
from huggingface_hub import hf_hub_download
import timm
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()
out = {}
for ad in ('vit_base_patch14_dinov2.lvd142m', 'vit_small_patch14_dinov2.lvd142m'):
    p = hf_hub_download('timm/' + ad, 'model.safetensors')
    timm.create_model(ad, pretrained=True, img_size=224, num_classes=0)
    out['timm/' + ad] = {'sha256': sha(p), 'yol': p}
w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
torchvision.models.resnet18(weights=w)
import os, pathlib
p = pathlib.Path(os.environ['TORCH_HOME']) / 'hub' / 'checkpoints' / pathlib.Path(w.url).name
out['torchvision/' + p.name] = {'sha256': sha(p), 'yol': str(p)}
print(json.dumps(out))
'''
os.environ.update({'HF_HOME': str(IS / 'data' / 'hf_cache'), 'TORCH_HOME': str(IS / 'data' / 'torch_cache')})
r = subprocess.run(['python', '-c', AG_KOD], capture_output=True, text=True, cwd=IS)
assert r.returncode == 0, r.stderr[-3000:]
ag = json.loads(r.stdout.strip().splitlines()[-1])
beklenen = json.loads((IS / 'colab' / 'veri_listesi.json').read_text(encoding='utf-8'))['agirliklar']
for k, v in ag.items():
    assert v['sha256'] == beklenen[k]['sha256'], f'{k}: SHA256 {v["sha256"]} ≠ yerel {beklenen[k]["sha256"]} (farklı önağırlık → DUR)'
    print('OK', k, v['sha256'][:12])
os.environ['HF_HUB_OFFLINE'] = '1'   # kuyruk yalnız doğrulanmış önbelleği kullanır

OZ = IS / 'oz'; shutil.rmtree(OZ, ignore_errors=True); OZ.mkdir()
def tf32(etiket, onek, env, ek=()):
    r = subprocess.run(onek + ['colab/oz_sinama.py', '--tf32', '--etiket', etiket] + list(ek),
                       capture_output=True, text=True, cwd=IS, env=env)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])
env_guc = {k: v for k, v in os.environ.items() if k != 'NVIDIA_TF32_OVERRIDE'}   # güç sınaması: ortam değişkeni de yok
T = {'sarmalayici': tf32('sarmalayici', ['python', 'colab/fp32_kos.py'], os.environ.copy()),
     'sarmalayici+seed_bagla': tf32('sarmalayici+seed_bagla', ['python', 'colab/fp32_kos.py'], os.environ.copy(), ['--seed-bagla']),
     'sarmalayici+seed_bagla_ortamsiz': tf32('sarmalayici+seed_bagla_ortamsiz', ['python', 'colab/fp32_kos.py'], env_guc, ['--seed-bagla']),
     'ortam_degiskeni_yalniz': tf32('ortam_degiskeni_yalniz', ['python'], os.environ.copy()),
     'guc_varsayilan_ortamsiz': tf32('guc_varsayilan_ortamsiz', ['python'], env_guc)}
for k, v in T.items():
    print(f"{k:<26} matmul {v['matmul_goreli_hata']:.2e} conv {v['conv_goreli_hata']:.2e} → {'IEEE FP32' if v['ieee'] else 'TF32/düşük hassasiyet'}")
assert T['sarmalayici']['ieee'] and T['sarmalayici+seed_bagla']['ieee'], 'TF32 KAPANMADI (ön-kayıt: FP32, TF32 kapalı) → DUR, şefe bildir'
guc = not T['guc_varsayilan_ortamsiz']['ieee']
print('sınamanın gücü:', 'TF32 varsayılan durumda GÖRÜLDÜ (sınama TF32 yakalayabiliyor)' if guc else 'varsayılan durum da IEEE çıktı (güç gösterilemedi; kayıt)')

def iki_kez(ad, arg):
    sonuc = []
    for i in (1, 2):
        out = OZ / f'{ad}_{i}' / 'metrics.json'
        rc = kos(['python', 'colab/fp32_kos.py'] + arg + ['--out', out], cwd=IS)
        assert rc == 0, f'öz-sınama {ad} rc={rc}'
        m = json.loads(out.read_text(encoding='utf-8'))
        sonuc.append((m.get('ilk_adim_loss'), (out.parent / 'tahmin_val.npy').read_bytes()))
    return {'ilk_adim_loss': [s[0] for s in sonuc], 'tahmin_bit_ayni': sonuc[0][1] == sonuc[1][1],
            'loss_ayni': sonuc[0][0] == sonuc[1][0]}
VIT = ['--lr-bas', '1e-3', '--lr-govde', '1e-5', '--wd', '0.01', '--katman-azalma', '1', '--isinma-epoch', '0', '--drop-path', '0']
D = {'r18': iki_kez('r18', ['varyantlar/r18_imagenet/ince_ayar.py', '--seed', '0', '--epoch', '1', '--alt-kume', '64', '--azami-adim', '2']),
     'vit_s_tam': iki_kez('vit_s_tam', ['varyantlar/dinov2_ft/ince_ayar_vit.py', '--mod', 'tam', '--model', 's', '--grad-ckpt', '0',
                                        '--mikro-bs', '16', '--seed', '0', '--epoch', '1', '--alt-kume', '16', '--azami-adim', '2'] + VIT)}
print(json.dumps(D, indent=1))
deterministik = all(v['tahmin_bit_ayni'] and v['loss_ayni'] for v in D.values())
(DRIVE / 'env' / f'oz-sinama-{OTURUM}.json').write_text(json.dumps(
    {'oturum': OTURUM, 'agirliklar': ag, 'tf32': T, 'tf32_guc': guc, 'determinizm': D, 'deterministik': deterministik},
    indent=1, ensure_ascii=False), encoding='utf-8')
if not deterministik:
    print('!!! UYARI: aynı seed iki kez bit bit aynı DEĞİL (kayıt env/oz-sinama-*.json). Ölçüm geçerliliğini bozmaz '
          '(kararlar 5-seed ortalamasıyla); şef değerlendirir.')
print('ÖZ-SINAMA bitti · deterministik:', deterministik)
"""

C7 = r"""# 7) CANLI KALP ATIŞI: her 60 sn yz/3-1/durum/canli.json (aktif koşum, epoch, val_acc, GPU bellek, son kalp atışı)
CANLI_IPLIK, CANLI_DUR = DY.canli_baslat(IS, DURUM, KUYRUK, OTURUM, aralik=60)
time.sleep(2)
print((DURUM / 'canli.json').read_text(encoding='utf-8')[:600])
"""

C8 = r"""# 8) KUYRUK — kademeli (duman → ön-uçuş → tam); aşama kapıları otomatik; oturum koptuysa DEVAM (bitmiş satırlar atlanır).
GRUP_IZNI = {'kc2-tur4': DRIVE / 'kuyruk' / 'kc2-tur4.izin.json'}   # şef tekrar kapısını geçince yazar
if KUYRUK in GRUP_IZNI:
    assert GRUP_IZNI[KUYRUK].exists(), (f'{KUYRUK} için şef izni yok ({GRUP_IZNI[KUYRUK].name}). Önce kc1-tekrar bitmeli ve '
                                         'tekrar kapısı şefte geçmeli. KUYRUK değişkenini kontrol et.')
    print('şef izni:', GRUP_IZNI[KUYRUK].read_text(encoding='utf-8')[:300])
os.chdir(IS)
t0 = time.time()
GRUP_OZET = DY.grup_kos(IS, KUYRUK, durda_devam=DURDA_DEVAM)
print(f'\nGRUP {KUYRUK} bitti · {(time.time() - t0) / 3600:.2f} sa')
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'aile'} for k, v in GRUP_OZET.items()}, indent=1, ensure_ascii=False))
"""

C9 = r"""# 9) SON: damga durumu + kuyruk listesi + aşama-3 beklenti raporu (bilgi) + leaderboard (+ kc1: replikasyon ve tekrar özeti)
os.chdir(IS)
kos(['python', 'orkestra/kosum.py', '--kok', '.', 'durum'], cwd=IS)
kos(['python', 'orkestra/kuyruk.py', '--kok', '.', 'liste'], cwd=IS)
G = DY.gruplar(IS)[KUYRUK]
aileler = {ad: DY.aile(IS, ad) for ad in G}
for ad, adlar in aileler.items():
    for x in adlar:
        if DY.kuyruk_damgasi(IS, x) and any(int(j.get('asama', 3)) == 3 for j in DY.satirlar(IS, x)):
            kos(['python', 'orkestra/beklenti_kontrol.py', '--kok', '.', '--kuyruk', x, '--asama', '3'], cwd=IS)
if KUYRUK == 'kc1-tekrar':
    kos(['python', 'orkestra/leaderboard.py', '--kok', '.', '--metrik', 'val_acc', '--yon', 'max', '--baseline', 'baseline_colab',
         '--csv', 'sonuclar/leaderboard-kc1.csv', '--md', 'sonuclar/leaderboard-kc1.md'], cwd=IS)
    kos(['python', 'orkestra/replikasyon_kapisi.py', '--kok', '.', '--metrik', 'test_acc', '--yon', 'max', '--lit', '0.7338',
         '--lit-ust', '0.742', '--tolerans', '0.012', '--baseline', 'baseline_colab',
         '--kaynak', 'Yang 2023 T3; Doerrich 2025 S4; Medeiros 2026 T2 (Colab L4 tekrarı; karar şefte)'], cwd=IS)
    ozet = DY.tekrar_ozeti(IS, aileler['kc1-tekrar'])
    (IS / 'sonuclar' / 'tekrar-ozeti-colab.md').write_text(ozet, encoding='utf-8')
    print(ozet)
else:
    kos(['python', 'orkestra/leaderboard.py', '--kok', '.', '--metrik', 'val_acc', '--yon', 'max', '--baseline', 'h02_r18in_ft224_colab',
         '--csv', 'sonuclar/leaderboard-colab.csv', '--md', 'sonuclar/leaderboard-colab.md'], cwd=IS)
    print(DY.tekrar_ozeti(IS, [x for adlar in aileler.values() for x in adlar]).split('FP32 denetimi:')[-1])
print('DermaMNIST-C arka plan:', (DURUM / 'veri_c.json').read_text(encoding='utf-8')[:400] if (DURUM / 'veri_c.json').exists() else 'henüz yok',
      '· iplik canlı:', VERI_C_IPLIK.is_alive())
CANLI_DUR.set(); time.sleep(1)
son = DY.canli_durum(IS, KUYRUK, OTURUM); son['bitti'] = True
DY.atomik_json(DURUM / 'canli.json', son)
print('BİTTİ. Sonuçlar Drive\'da: yz/3-1/runs, yz/3-1/sonuclar. Şef bilgisayarından okuyacak; zip/indirme GEREKMEZ.')
"""


def hucre(tur, kaynak):
    satirlar = kaynak.strip("\n").split("\n")
    src = [s + "\n" for s in satirlar[:-1]] + [satirlar[-1]]
    h = {"cell_type": tur, "metadata": {}, "source": src}
    if tur == "code":
        h.update({"execution_count": None, "outputs": []})
    return h


def main():
    hucreler = [hucre("markdown", MD0), hucre("code", C1), hucre("code", C2), hucre("code", C3), hucre("code", C4),
                hucre("code", C5), hucre("code", C6), hucre("code", C7), hucre("code", C8), hucre("code", C9)]
    nb = {"nbformat": 4, "nbformat_minor": 0,
          "metadata": {"colab": {"provenance": [], "gpuType": "L4", "machine_shape": "hm"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                       "accelerator": "GPU"},
          "cells": hucreler}
    out = HAT / "3-1-colab.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    # sözdizimi denetimi: her kod hücresi derlenebilir mi (get_ipython hariç çalıştırılmaz)
    for i, h in enumerate(hucreler):
        if h["cell_type"] == "code":
            compile("".join(h["source"]), f"hücre{i}", "exec")
    print(f"yazıldı: {out} ({len(hucreler)} hücre)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
