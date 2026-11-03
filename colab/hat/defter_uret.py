#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""defter_uret.py — v2 PARALEL SEKME defterlerini üretir: 3-1-colab-<SEKME>.ipynb (A, B, C …).

Bütün defterler AYNI koddur; YALNIZ ilk kod hücresi (SEKME, KUYRUK) farklıdır (üretimde denetlenir). KUYRUK tek kuyruk adı
(paralel: her sekme bir kuyruk) ya da kuyruk/gruplar.json'daki bir grup (tek sekmede sırayla) olabilir → K5'te de
aynı defter 2–3 kuyrukla `--sekmeler` değiştirilerek üretilir.
v1 tek defter (3-1-colab.ipynb, kc1-tekrar oturumu) → defter_uret_v1.py (dondurulmuş; hücre 2 DONANIM ve hücre 6
ÖNAĞIRLIK/ÖZ-SINAMA metinleri oradan BİREBİR alınır).

Kullanım: <uv-python> colab/hat/defter_uret.py [--sekmeler A=kc2-tur4-h10,B=kc2-tur4-h11,C=kc2-tur4-h12]
          → colab/hat/3-1-colab-A.ipynb … (drive_paketle.py Drive'a kopyalar: yz/3-1-colab-A.ipynb …).
K5 (2026-10-11): --k5 → K5 başlık hücresi (test kilidi / şef izni / en çok 2 eşzamanlı oturum); --dosya SEKME=ek,… → dosya adı
  3-1-colab-<ek>.ipynb (SEKME harf/rakam kalır). Kod hücreleri (2–10, hücre 8 yarış düzeltmesi dahil) v2 ile BİREBİR aynıdır:
  <uv-python> colab/hat/defter_uret.py --k5 --sekmeler K5T=k5-test --dosya K5T=K5-test
  <uv-python> colab/hat/defter_uret.py --k5 --sekmeler C1A=k5-c1-A,C1B=k5-c1-B --dosya C1A=K5-C1-A,C1B=K5-C1-B
  <uv-python> colab/hat/defter_uret.py --k5 --sekmeler C2=k5-c2 --dosya C2=K5-C2"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HAT = Path(__file__).resolve().parent
sys.path.insert(0, str(HAT))
import defter_uret_v1 as V1  # noqa: E402

VARSAYILAN_SEKMELER = "A=kc2-tur4-h10,B=kc2-tur4-h11,C=kc2-tur4-h12"

MD0 = r"""# 3.1 DermaMNIST — Colab hattı v2 (PARALEL SEKMELER)

**Ne yapar:** Drive'daki kod paketini (`yz/3-1/`) doğrular, veriyi indirir, ortamı sınar ve ilk hücredeki **KUYRUK**'u
kademeli koşar: duman → ön-uçuş → tam. Aşama kapıları otomatiktir. Beklenti ihlalinde tam koşum **başlamaz**.

Bu defterin kardeşleri (`3-1-colab-A/B/C.ipynb`) **aynı koddur**; yalnız ilk hücre (SEKME, KUYRUK) farklıdır. Her biri
ayrı bir Colab oturumunda (ayrı L4) aynı anda koşar. Şef hangi sekmenin hangi kuyruğu koşacağını ilk hücreye yazmıştır.

**Senin yapacağın (her sekme için):** *Çalışma zamanı → Çalışma zamanı türünü değiştir → **L4 GPU*** → *Tümünü çalıştır* →
sekmeyi açık tut. Bir sekme koparsa yalnız O defteri yeniden aç, yine *Tümünü çalıştır* (≈ 10 dk bekle; aynı kuyruğu iki
sekmenin koşması engellenir). Devam mantığı bitmiş koşumları atlar.

Drive'a yazılan ortak dosyalar oturuma ayrılır: `durum/canli_<oturum>.json` (60 sn kalp atışı), `durum/hata_<oturum>_*.log`,
`runs/BILDIRIM_<oturum>.log`, `env/*-<oturum>.*`. Kuyruk damgaları kuyruk adına göre zaten ayrıktır. Şef hepsini kendi
bilgisayarından izler.
"""

MD0_K5 = r"""# 3.1 DermaMNIST — K5 Colab defteri (Colab hattı v2; kod hücreleri Tur-4 sekmeleriyle AYNI)

**Ne yapar:** Drive'daki kod paketini (`yz/3-1/`) doğrular, veriyi indirir, ortamı sınar ve ilk hücredeki **KUYRUK**'u kademeli
koşar. Aşama kapıları otomatiktir; beklenti ihlalinde sonraki aşama **başlamaz**.

**K5 kuralları (hipotezler/K5-ONKAYIT.md):**
- Her K5 kuyruğu **şef izni** ister (`kuyruk/<grup>.izin.json`). İzin yoksa hücre 5 durur. İzni yalnız şef yazar.
- `k5-test`: a1 = yükleme özdeşliği (yalnız val; test açılmaz), a3 = resmî test çıkarımı (**bir kez**). Bu defter test
  sayısı **göstermez**; sayılar şefin `k5_test.py` analizinde bir kez hesaplanır.
- C1 (`k5-c1-A`, `k5-c1-B`) **resmî test geçişinden SONRA** açılır; C2 (`k5-c2`) C1 bitince.
- **Aynı anda en çok 2 Colab oturumu** (kullanıcı hesabı sınırı). Test sekmesi açıkken en çok 1 ek sekme.

**Senin yapacağın:** *Çalışma zamanı → Çalışma zamanı türünü değiştir → **L4 GPU*** → *Tümünü çalıştır* → sekmeyi açık tut.
Sekme koparsa yalnız O defteri yeniden aç, yine *Tümünü çalıştır* (≈ 10 dk bekle). Bitmiş koşumlar atlanır.
Bitince şefe "bitti" demen yeterli; dosya indirmen gerekmez.
"""

C1 = r"""# 1) AYARLAR — sekmeler arasında YALNIZ bu hücre farklıdır. Şef yazar; DEĞİŞTİRME.
SEKME = '{sekme}'                # oturum kimliğinin başı (canlı dosya, hata günlüğü, bildirim dosyası adlarında)
KUYRUK = '{kuyruk}'      # tek kuyruk adı (paralel sekme) YA DA kuyruk/gruplar.json'daki bir grup (tek sekmede sırayla)
DURDA_DEVAM = False              # (grup kipinde) bir kuyruk aşama kapısında DURURSA kalan kuyruklar koşmaz
"""

C2 = r"""# 2) KURULUM: Drive + oturum kimliği + kuyruk çözümü + hata günlüğü (bütün sekmelerde aynı)
import os, sys, json, time, shutil, subprocess, traceback, pathlib, datetime, hashlib, secrets, re
os.environ['TZ'] = 'Europe/Istanbul'; time.tzset()   # run_id ve damgalar yerel saatle (yerel koşumlarla kıyaslanır)
from google.colab import drive
drive.mount('/content/drive')
YZ = pathlib.Path('<DRIVE_KOK>')
DRIVE = YZ / '3-1'              # kalıcı çalışma kökü (kod paketi + runs/ kuyruk/ sonuclar/ env/ durum/)
IS = pathlib.Path('/content/is')    # oturumluk çalışma kopyası (kod Drive'dan kopyalanır, SHA256 doğrulanır)
VERI = pathlib.Path('/content/veri')  # oturumluk veri (Zenodo'dan indirilir, MD5 + SHA256)
DURUM = DRIVE / 'durum'
DURUM.mkdir(parents=True, exist_ok=True)
assert re.fullmatch(r'[A-Za-z0-9]{1,8}', SEKME), f'SEKME harf/rakam olmalı: {SEKME!r}'
# Oturum kimliği: sekme + başlangıç zamanı + rastgele ek → paralel/yeniden açılan oturumlar asla aynı adı almaz
OTURUM = f"{SEKME}-{datetime.datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"

def _hata_kaydet(shell, etype, value, tb, tb_offset=None):
    # Her hücre hatası: tam traceback → yz/3-1/durum/hata_<oturum>_<zaman>.log (şefin yerel izleyicisi HATA-LOG basar)
    satirlar = traceback.format_exception(etype, value, tb)
    try:
        hucre = (shell.user_ns.get('In') or [''])[-1]
        yol = DURUM / f'hata_{OTURUM}_{datetime.datetime.now():%Y%m%d-%H%M%S}.log'
        yol.write_text(f'oturum {OTURUM} · sekme {SEKME} · kuyruk {KUYRUK}\n--- hücre ---\n{hucre}\n--- traceback ---\n'
                       + ''.join(satirlar), encoding='utf-8')
        satirlar = satirlar + [f'\n[hata günlüğü] {yol}\n']
    except Exception as e:
        satirlar = satirlar + [f'\n[hata günlüğü YAZILAMADI: {e}]\n']
    return satirlar
get_ipython().set_custom_exc((Exception,), _hata_kaydet)

assert (DRIVE / 'PAKET.json').exists(), (f'{DRIVE}/PAKET.json yok. Drive eşitlemesi bitmemiş olabilir; '
                                        'birkaç dakika bekleyip defteri yeniden çalıştır.')
_gr = json.loads((DRIVE / 'kuyruk' / 'gruplar.json').read_text(encoding='utf-8'))['gruplar']
if KUYRUK in _gr:
    KUYRUKLAR = list(_gr[KUYRUK])
elif (DRIVE / 'kuyruk' / f'{KUYRUK}.jsonl').exists() and not re.search(r'-devam\d+$', KUYRUK):
    KUYRUKLAR = [KUYRUK]
else:
    raise AssertionError(f'KUYRUK {KUYRUK!r} ne grup ne kuyruk dosyası (kuyruk/{KUYRUK}.jsonl). İlk hücreyi şefe sor.')

def kos(cmd, cwd=None, env=None):
    # Komutu koşar, çıktıyı canlı basar, dönüş kodunu verir
    cmd = [str(c) for c in cmd]
    print('$', ' '.join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
    for satir in p.stdout:
        print(satir, end='', flush=True)
    return p.wait()

print('oturum', OTURUM, '· sekme', SEKME, '· KUYRUK', KUYRUK, '→', KUYRUKLAR, '· Drive', DRIVE)
"""

C3 = V1.C2.replace("# 2) DONANIM", "# 3) DONANIM", 1)

C4 = r"""# 4) ORTAM: Colab'ın kendi torch'u (pin YOK; sürümler kaydedilir) + medmnist/timm/tensorboardX pinleri.
#    TF32 kapalı + deterministik ortam değişkenleri (tüm alt süreçler devralır). Defter çekirdeği torch'u İÇE AKTARMAZ.
#    Kayıtlar oturum başına ayrı dosyaya (paralel sekmeler aynı Drive dosyasına yazmaz).
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
ortam.update({'zaman': datetime.datetime.now().isoformat(timespec='seconds'), 'oturum': OTURUM, 'sekme': SEKME,
              'kuyruk': KUYRUK, 'kuyruklar': KUYRUKLAR, 'nvidia_smi': gpu,
              'ortam_degiskenleri': {k: os.environ[k] for k in ('NVIDIA_TF32_OVERRIDE', 'CUBLAS_WORKSPACE_CONFIG', 'TZ')}})
ENV = DRIVE / 'env'; ENV.mkdir(exist_ok=True)
(ENV / f'ortam-colab-{OTURUM}.json').write_text(json.dumps(ortam, indent=1, ensure_ascii=False), encoding='utf-8')
(ENV / f'requirements-colab-{OTURUM}.lock').write_text(
    subprocess.run(['python', '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout, encoding='utf-8')
# Sürüm sürüklenmesi: ilk Colab oturumuna göre (env/ortam-colab-ilk.json; kc1 oturumu yazdı, salt okunur).
# Başlamış bir kuyruğun DEVAMINDA çekirdek sürüm değiştiyse DUR.
ILK = ENV / 'ortam-colab-ilk.json'
CEKIRDEK = ('python', 'torch', 'cuda', 'cudnn', 'torchvision', 'numpy', 'timm', 'medmnist', 'sklearn')
if not ILK.exists():
    ILK.write_text(json.dumps(ortam, indent=1, ensure_ascii=False), encoding='utf-8'); print('ilk Colab ortamı kaydedildi')
else:
    ilk = json.loads(ILK.read_text(encoding='utf-8'))
    fark = {k: (ilk.get(k), ortam.get(k)) for k in CEKIRDEK if ilk.get(k) != ortam.get(k)}
    if fark:
        print('!!! UYARI: Colab ortamı ilk oturumdan FARKLI:', fark)
        baslamis = [p.name for ad in KUYRUKLAR for p in (DRIVE / 'runs' / 'kuyruklar').glob(f'{ad}*.baslat.json')] \
            if (DRIVE / 'runs' / 'kuyruklar').exists() else []
        assert not baslamis, (f'Başlamış kuyruk {baslamis} farklı çekirdek sürümle sürdürülemez (seed\'ler iki ortama bölünür). '
                              f'DUR → şefe bildir. Fark: {fark}')
    else:
        print('ortam ilk Colab oturumuyla aynı (çekirdek sürümler)')
"""

C5 = r"""# 5) PAKET BÜTÜNLÜĞÜ + İZİN + ÇAKIŞMA: Drive'daki kod → /content/is (kopya), her dosya PAKET.json SHA256'sıyla doğrulanır.
#    runs/ kuyruk/ sonuclar/ env/ → Drive'a bağ (kalıcı); data/medmnist, data/dermamnist_c → /content/veri.
paket = json.loads((DRIVE / 'PAKET.json').read_text(encoding='utf-8'))
yardim_rel = 'colab/defter_yardim.py'
gecici = pathlib.Path('/tmp/defter_yardim_dogrula.py')
shutil.copyfile(DRIVE / yardim_rel, gecici)
assert hashlib.sha256(gecici.read_bytes()).hexdigest() == paket['dosyalar'][yardim_rel]['sha256'], \
    'defter_yardim.py SHA256 PAKET.json ile uyuşmuyor (Drive eşitlemesi yarım?) — DUR'
IS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, '/tmp'); import importlib; DY = importlib.import_module('defter_yardim_dogrula')
assert hasattr(DY, 'es_zamanli_cakisma'), f"Drive'daki paket v2 değil (sürüm {paket.get('surum')}): paralel sekme desteği yok — şefe bildir"
pk = DY.paket_kopyala_dogrula(DRIVE, IS)
print(json.dumps({k: v for k, v in pk.items() if k != 'fazla'}, ensure_ascii=False))
assert not pk['bozuk'] and not pk['eksik'], f"PAKET doğrulaması BAŞARISIZ: bozuk {pk['bozuk'][:5]} eksik {pk['eksik'][:5]}"
if pk['fazla']:
    print('UYARI: pakette olmayan dosyalar (kullanılmaz):', pk['fazla'][:10])
for rel, b in paket.get('kuyruk_dosyalari', {}).items():   # ön-kayıtlı kuyruk + beklenti dosyaları Drive'da bayt bayt aynı mı
    assert hashlib.sha256((DRIVE / rel).read_bytes()).hexdigest() == b['sha256'], \
        f'{rel} PAKET.json ile uyuşmuyor (ön-kayıt değişmiş ya da Drive eşitlemesi yarım) — DUR, şefe bildir'
print('kuyruk/beklenti dosyaları doğrulandı:', len(paket.get('kuyruk_dosyalari', {})))
# KUYRUK çözümü + şef izni (tek kuyruk, grubunun iznini devralır: kc2-tur4-h10 → kuyruk/kc2-tur4.izin.json)
assert DY.kuyruk_cozumle(DRIVE, KUYRUK) == KUYRUKLAR, (DY.kuyruk_cozumle(DRIVE, KUYRUK), KUYRUKLAR)
IZIN = DY.izin_dosyasi(DRIVE, KUYRUK)
if IZIN is not None:
    assert IZIN.exists(), (f'{KUYRUK} için şef izni yok ({IZIN.name}). Önce tekrar kapısı şefte geçmeli. '
                           'İlk hücreyi kontrol et; şefe bildir.')
    print('şef izni:', IZIN.read_text(encoding='utf-8')[:300])
# Aynı kuyruğu başka bir sekme koşuyor mu? (taze kalp atışı + kuyruk kesişimi → DUR)
CAKISMA = DY.es_zamanli_cakisma(DRIVE, DURUM, KUYRUKLAR, OTURUM)
assert not CAKISMA, (f'Bu kuyruk başka bir sekmede KOŞUYOR: {CAKISMA}. Aynı kuyruğu iki sekme koşamaz. Diğer sekme kapandıysa '
                     '≈ 10 dk bekleyip yeniden çalıştır; emin değilsen şefe sor.')
bag = DY.baglantilari_kur(DRIVE, IS, VERI)
print('bağlar:', json.dumps(bag, ensure_ascii=False))
(DRIVE / 'env' / f'paket-{OTURUM}.json').write_text(json.dumps(
    {'oturum': OTURUM, 'sekme': SEKME, 'kuyruk': KUYRUK, 'kuyruklar': KUYRUKLAR, 'surum': pk['surum'], 'dosya': pk['kopyalanan'],
     'zaman': datetime.datetime.now().isoformat(timespec='seconds')}, ensure_ascii=False, indent=1), encoding='utf-8')
os.chdir(IS)
print('PAKET OK · sürüm', pk['surum'], '· çalışma kökü', os.getcwd())
"""

C6 = r"""# 6) VERİ: Zenodo → /content/veri (oturum başına), resmî MD5 + yerel SHA256 + bayt. 64/128 gerekmez.
#    DermaMNIST-C (28/224) arka planda iner (kuyruk beklemez; K5 içindir) → durum/veri_c_<oturum>.json.
vm = DY.veri_hazirla(IS / 'colab' / 'veri_listesi.json', VERI, 'medmnist', DURUM / f'veri_medmnist_{OTURUM}.json')
assert all(v['dogrulandi'] for v in vm.values()), vm
VERI_C_JSON = DURUM / f'veri_c_{OTURUM}.json'
VERI_C_IPLIK = DY.veri_arka_plan(IS / 'colab' / 'veri_listesi.json', VERI, 'dermamnist_c', VERI_C_JSON,
                                 DURUM / f'veri_c_{OTURUM}.log')
print('MedMNIST 28/224 doğrulandı; DermaMNIST-C arka planda →', VERI_C_JSON)
"""

C7 = V1.C6.replace("# 6) ÖNAĞIRLIKLAR", "# 7) ÖNAĞIRLIKLAR", 1)

C8 = r"""# 8) CANLI KALP ATIŞI: her 60 sn yz/3-1/durum/canli_<oturum>.json (yalnız bu sekmenin kuyruk ailelerinin aktif koşumları,
#    epoch, val_acc, GPU bellek, son kalp atışı). Paralel sekmeler ayrı dosyaya yazar.
CANLI_IPLIK, CANLI_DUR = DY.canli_baslat(IS, DURUM, KUYRUK, OTURUM, aralik=60, oturum_dosyasi=True, kuyruklar=KUYRUKLAR,
                                         sekme=SEKME)
CANLI = DY.canli_dosyasi(DURUM, OTURUM)
for _ in range(90):   # Drive yazımı gecikebilir (2026-10-11: 2 sn yetmedi → hücre durdu); en çok 90 sn bekle, hata verme
    if CANLI.exists(): break
    time.sleep(1)
print(CANLI.name, CANLI.read_text(encoding='utf-8')[:600] if CANLI.exists() else '(henüz yazılmadı — iş parçacığı çalışıyor; devam)')
"""

C9 = r"""# 9) KUYRUK — kademeli (duman → ön-uçuş → tam); aşama kapıları otomatik; oturum koptuysa DEVAM (bitmiş satırlar atlanır).
#    Bildirimler runs/BILDIRIM_<oturum>.log'a (colab/oturum_sar.py; kuyruk.py/kosum.py kodu değişmez).
if IZIN is not None:
    assert IZIN.exists(), f'şef izni ({IZIN.name}) kayboldu — DUR, şefe bildir'
CAKISMA = DY.es_zamanli_cakisma(IS, DURUM, KUYRUKLAR, OTURUM)
assert not CAKISMA, f'Bu kuyruk başka bir sekmede KOŞUYOR: {CAKISMA} — DUR (aynı kuyruğu iki sekme koşamaz)'
os.chdir(IS)
t0 = time.time()
GRUP_OZET = DY.grup_kos(IS, KUYRUK, durda_devam=DURDA_DEVAM, oturum=OTURUM)
print(f'\nKUYRUK {KUYRUK} bitti · {(time.time() - t0) / 3600:.2f} sa')
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'aile'} for k, v in GRUP_OZET.items()}, indent=1, ensure_ascii=False))
"""

C10 = r"""# 10) SON: damga durumu + kuyruk listesi + aşama-3 beklenti raporu (bilgi) + leaderboard (kuyruğa özgü dosya)
#     Not: 'kosum.py durum' diğer sekmelerin koşan koşumlarını 'ÖLDÜ damgasız' gösterebilir (PID'leri bu VM'de yok) —
#     yalnız gösterim; damgalama YAPILMAZ.
os.chdir(IS)
kos(['python', 'orkestra/kosum.py', '--kok', '.', 'durum'], cwd=IS)
kos(['python', 'orkestra/kuyruk.py', '--kok', '.', 'liste'], cwd=IS)
aileler = {ad: DY.aile(IS, ad) for ad in KUYRUKLAR}
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
         '--csv', f'sonuclar/leaderboard-colab-{KUYRUK}.csv', '--md', f'sonuclar/leaderboard-colab-{KUYRUK}.md'], cwd=IS)
    print(DY.tekrar_ozeti(IS, [x for adlar in aileler.values() for x in adlar]).split('FP32 denetimi:')[-1])
print('DermaMNIST-C arka plan:', VERI_C_JSON.read_text(encoding='utf-8')[:400] if VERI_C_JSON.exists() else 'henüz yok',
      '· iplik canlı:', VERI_C_IPLIK.is_alive())
CANLI_DUR.set(); time.sleep(1)
son = DY.canli_durum(IS, KUYRUK, OTURUM, KUYRUKLAR, SEKME); son['bitti'] = True
DY.atomik_json(CANLI, son)
print(f'BİTTİ (sekme {SEKME}, {KUYRUK}). Sonuçlar Drive\'da: yz/3-1/runs, yz/3-1/sonuclar. Şef bilgisayarından okuyacak; '
      'zip/indirme GEREKMEZ. Bu sekmeyi kapatabilirsin.')
"""


def defter(sekme: str, kuyruk: str, k5: bool = False) -> dict:
    hucreler = [V1.hucre("markdown", MD0_K5 if k5 else MD0), V1.hucre("code", C1.format(sekme=sekme, kuyruk=kuyruk))] + \
               [V1.hucre("code", c) for c in (C2, C3, C4, C5, C6, C7, C8, C9, C10)]
    return {"nbformat": 4, "nbformat_minor": 0,
            "metadata": {"colab": {"provenance": [], "gpuType": "L4", "machine_shape": "hm"},
                         "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                         "accelerator": "GPU"},
            "cells": hucreler}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sekmeler", default=VARSAYILAN_SEKMELER, help="SEKME=KUYRUK,… (KUYRUK tek kuyruk ya da grup)")
    ap.add_argument("--k5", action="store_true", help="K5 başlık hücresi")
    ap.add_argument("--dosya", default="", help="SEKME=dosya_eki,… → 3-1-colab-<ek>.ipynb (varsayılan ek = SEKME)")
    a = ap.parse_args()
    dosya_ek = dict(x.split("=", 1) for x in a.dosya.split(",") if x.strip())
    for e in dosya_ek.values():
        assert re.fullmatch(r"[A-Za-z0-9-]{1,24}", e), e
    eslem = []
    for parca in a.sekmeler.split(","):
        s, q = parca.split("=", 1)
        s, q = s.strip(), q.strip()
        assert re.fullmatch(r"[A-Za-z0-9]{1,8}", s), s
        assert re.fullmatch(r"[A-Za-z0-9_.-]+", q) and "'" not in q, q
        eslem.append((s, q))
    assert len({s for s, _ in eslem}) == len(eslem), "SEKME adları benzersiz olmalı"
    assert len({q for _, q in eslem}) == len(eslem), "aynı KUYRUK iki sekmeye verilemez"
    nbs = {}
    for s, q in eslem:
        nb = defter(s, q, a.k5)
        for i, h in enumerate(nb["cells"]):  # sözdizimi denetimi (çalıştırılmaz)
            if h["cell_type"] == "code":
                compile("".join(h["source"]), f"{s}-hücre{i}", "exec")
        out = HAT / f"3-1-colab-{dosya_ek.get(s, s)}.ipynb"
        out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        nbs[s] = nb
        print(f"yazıldı: {out} (SEKME={s}, KUYRUK={q}, {len(nb['cells'])} hücre)")
    # denetim: defterler yalnız ilk KOD hücresinde (indeks 1) farklı
    adlar = list(nbs)
    for s in adlar[1:]:
        a0, b0 = nbs[adlar[0]]["cells"], nbs[s]["cells"]
        fark = [i for i, (x, y) in enumerate(zip(a0, b0)) if x != y]
        assert len(a0) == len(b0) and fark == [1], (s, fark)
    print(f"denetim: {len(adlar)} defter yalnız hücre 1'de (AYARLAR) farklı ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
