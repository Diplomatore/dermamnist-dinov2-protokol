#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""defter_uret_e1.py — E1 (H14) Colab defteri `colab/e1-h14.ipynb`'yi üretir (hipotezler/H14.md; KULLANICI ONAYI 2026-11-03).

Hat hücreleri yeniden kullanılır (defter_uret.py v2: kurulum/paket/veri/canlı; defter_uret_v1.py: donanım + önağırlık/öz-sınama).
E1'e özgü olanlar:
  hücre 1  PARAMETRELER — kişisel yol (DRIVE_YZ), KUYRUK (e1-h14 | e1-h10r), Durum B seçeneği, torch sabitleme hedefi.
  hücre 2  kurulum: v2 C2, yalnız Drive yolu parametreden (YZ = DRIVE_YZ).
  hücre 4  ORTAM: torch'u Ekim sürümüne (2.11.0+cu130) SABİTLEMEYİ DENER (kart "Ortam"); olmazsa mevcut sürümle devam
           (Durum B yolu; kayıt). Sürüklenme denetimi E1'in ilk oturumuna göre (env/ortam-colab-e1-ilk.json).
  hücre 8  E1 HAZIRLIK: disk (kural b ön denetimi), referans koşum (h10_c1_duman_s0) Drive'da mı, oturum koptuysa C1 verisi
           (devam kuyruğunda veri satırı yoksa), e1-h10r yalnız kontrol kolu Durum B ise.
  hücre 10 KUYRUK + Durum izleyici (kontrol kolu bitince durum/e1_durum_<oturum>.json).
  hücre 11 DURUM B DALI: Durum B ise e1-h10r (aynı oturumda sırayla ya da ikinci sekmede).
  hücre 12 SON: damgalar, a3 kapı raporları, E1 bütçe, Drive'da hangi klasörlerin şefe gideceği. Test sayısı GÖSTERİLMEZ.
Kullanım: <uv-python> colab/hat/defter_uret_e1.py  → colab/e1-h14.ipynb (drive_paketle.py Drive'a yz/e1-h14.ipynb olarak kopyalar)."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HAT = Path(__file__).resolve().parent
sys.path.insert(0, str(HAT))
import defter_uret as V2  # noqa: E402
import defter_uret_v1 as V1  # noqa: E402

WS = HAT.parents[1]
CIKTI = WS / "colab" / "e1-h14.ipynb"

MD0 = r"""# 3.1 DermaMNIST — E1 / H14 Colab defteri (test sonrası keşifsel; KULLANICI ONAYI 2026-11-03)

**Ne yapar:** H10'un C1 tarifine eğitim-zamanı D4 ekleyen kolu (H14) Colab L4'te koşar. Kod paketini Drive'dan (`yz/3-1/`)
doğrular, veriyi indirir, ortamı sınar ve **KUYRUK**'u kademeli koşar: a1 duman (kontrol kolu + D4 dumanı) → a2 ön-uçuş
(2 seed × 5 ep) → a3 tam (5 seed × 15 ep) + C-test çıkarımı (**bir kez**). Aşama kapıları otomatiktir. Beklenti ihlalinde
sonraki aşama **başlamaz**.

**Kurallar (hipotezler/H14.md):**
- Kontrol kolu (`h14_c1_d4kapali_duman`) Ekim'deki `h10_c1_duman_s0`'ı bit bit üretmelidir (**Durum A**). Sürüm farkı
  varsa **Durum B**: aynı ortamda H10 yeniden koşulur (`e1-h10r`; bu defter kendisi yapar). **Durum C** → DUR (şefe bildir).
- Resmî test **açılmaz**. C1 modelleri yalnız C-test'te (1.227) bir kez değerlendirilir. Bu defter test sayısı **göstermez**.
- E1 bütçesi 5,0 L4-sa; aşılırsa satırlar kendiliğinden başlamaz (rc=94).
- 8 görünümlü önbellek ≈ 26 GB, **Colab yerel diskine** (`/content`) yazılır; Drive'a değil.

**Senin yapacağın:** yalnız ilk hücredeki `DRIVE_YZ` yolunu kontrol et → *Çalışma zamanı → Çalışma zamanı türünü
değiştir → **L4 GPU*** → *Tümünü çalıştır* → sekmeyi açık tut. Ayrıntı: `colab/E1-OKU-BENI.md`.
"""

C1 = r"""# 1) PARAMETRELER — kişiye özel yollar ve E1 seçenekleri (yalnız bu hücreyi düzenleyebilirsin)
DRIVE_YZ = '<DRIVE_KOK>'   # Drive'daki 'yz' klasörü (Drive for desktop'ta: <DRIVE_KOK>). Farklıysa düzelt.
KUYRUK = 'e1-h14'          # 'e1-h14' (ANA; önce bu) | 'e1-h10r' (yalnız Durum B'de, İKİNCİ sekmede paralel koşturmak istersen)
H10R_AYNI_OTURUM = True    # Durum B'de e1-h14 bitince e1-h10r bu sekmede sırayla koşsun (False → ikinci sekmede elle)
DURDA_DEVAM = False        # bir aşama kapısı DURURSA devam edilmez (şef teşhis eder)
TORCH_PIN = '2.11.0+cu130'           # Ekim (H10 C1) sürümü — Durum A olasılığı için sabitlenmeye ÇALIŞILIR
TORCHVISION_PIN = '0.26.0+cu130'
PIN_INDEX = 'https://download.pytorch.org/whl/cu130'
EKIM_ORTAM = {'torch': '2.11.0+cu130', 'cuda': '13.0', 'cudnn': 92700, 'timm': '1.0.30'}   # h10_c1_duman_s0 config.json
DISK_MIN_GB = 40           # H14 kural (b): 8 görünüm önbelleği (26,2 GB) yazılmadan önce boş disk alt sınırı
REF_RUN = '20261011-215538_h10_c1_duman_s0'   # kural (f) referansı (Drive yz/3-1/runs altında)
SEKME = 'E1' if KUYRUK == 'e1-h14' else 'E1R'
assert KUYRUK in ('e1-h14', 'e1-h10r'), f'KUYRUK yalnız e1-h14 ya da e1-h10r olabilir: {KUYRUK!r}'
"""

C2 = V2.C2.replace("YZ = pathlib.Path('<DRIVE_KOK>')", "YZ = pathlib.Path(DRIVE_YZ)   # hücre 1 parametresi", 1)
assert C2 != V2.C2, "defter_uret.C2 Drive yolu satırı bulunamadı"

C3 = V1.C2.replace("# 2) DONANIM", "# 3) DONANIM", 1)

C4 = r"""# 4) ORTAM (E1): torch'u Ekim sürümüne SABİTLEMEYİ DENE (H14 "Ortam": Durum A olasılığı) → olmazsa mevcut sürümle devam
#    (Durum B yolu; sürümler kaydedilir). medmnist/timm/tensorboardX pinleri hat ile aynı. Defter çekirdeği torch'u İÇE AKTARMAZ
#    (pip sonrası yeniden başlatma gerekmez; kuyruk alt süreçleri yeni kurulumu kullanır).
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
PIN = {'hedef': TORCH_PIN, 'colab_torch': once['torch'], 'colab_torchvision': once['torchvision'], 'denendi': False}
if once['torch'] != TORCH_PIN:
    PIN['denendi'] = True
    PIN['rc'] = kos(['python', '-m', 'pip', 'install', '-q', f'torch=={TORCH_PIN}', f'torchvision=={TORCHVISION_PIN}',
                     '--index-url', PIN_INDEX])
    ara = surumler()
    if not (ara['cuda_var'] and 'L4' in str(ara['cihaz'])):
        print('!!! sabitlenen torch CUDA/L4 görmüyor → Colab sürümüne GERİ DÖNÜLÜYOR')
        ek = once['torch'].split('+')[1] if '+' in once['torch'] else 'cu130'
        kos(['python', '-m', 'pip', 'install', '-q', f"torch=={once['torch']}", f"torchvision=={once['torchvision']}",
             '--index-url', f'https://download.pytorch.org/whl/{ek}'])
        PIN['geri_donuldu'] = True
PIN['basarili'] = surumler()['torch'] == TORCH_PIN
print('torch sabitleme:', json.dumps(PIN, ensure_ascii=False))
ara = surumler()
assert kos(['python', '-m', 'pip', 'install', '-q', 'medmnist==3.0.2', 'timm==1.0.30', 'tensorboardX==2.6.5']) == 0, 'pip başarısız'
ortam = surumler()
print(json.dumps(ortam, indent=1, ensure_ascii=False))
assert ortam['torch'] == ara['torch'], f"pip torch'u değiştirdi: {ara['torch']} → {ortam['torch']} (DUR)"
for k, v in {'medmnist': '3.0.2', 'timm': '1.0.30', 'tensorboardX': '2.6.5'}.items():
    assert str(ortam[k]) == v, f'{k} pinli sürüm değil: {ortam[k]} ≠ {v}'
assert ortam['cuda_var'] and 'L4' in str(ortam['cihaz']), f'torch CUDA/L4 görmüyor: {ortam}'
EKIM_FARK = {k: [v, ortam.get(k)] for k, v in EKIM_ORTAM.items() if str(v) != str(ortam.get(k))}
print('Ekim (H10 C1) ortamına göre fark:', EKIM_FARK or 'YOK → Durum A beklenir',
      '' if not EKIM_FARK else '→ Durum B olası (kontrol kolu karar verir; e1-h10r gerekebilir)')

os.environ.update({'NVIDIA_TF32_OVERRIDE': '0', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONIOENCODING': 'utf-8',
                   'HF_HUB_DISABLE_TELEMETRY': '1'})
ortam.update({'zaman': datetime.datetime.now().isoformat(timespec='seconds'), 'oturum': OTURUM, 'sekme': SEKME,
              'kuyruk': KUYRUK, 'kuyruklar': KUYRUKLAR, 'nvidia_smi': gpu, 'torch_sabitleme': PIN, 'ekim_farki': EKIM_FARK,
              'ortam_degiskenleri': {k: os.environ[k] for k in ('NVIDIA_TF32_OVERRIDE', 'CUBLAS_WORKSPACE_CONFIG', 'TZ')}})
ENV = DRIVE / 'env'; ENV.mkdir(exist_ok=True)
(ENV / f'ortam-colab-{OTURUM}.json').write_text(json.dumps(ortam, indent=1, ensure_ascii=False), encoding='utf-8')
(ENV / f'requirements-colab-{OTURUM}.lock').write_text(
    subprocess.run(['python', '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout, encoding='utf-8')
# Sürüklenme: E1'in İLK oturumuna göre (env/ortam-colab-e1-ilk.json). Başlamış bir E1 kuyruğunun devamında ya da Durum B'de
# H10r'nin H14'ten farklı ortamda koşması → DUR (karşılaştırma aynı ortamda olmalı).
ILK = ENV / 'ortam-colab-e1-ilk.json'
CEKIRDEK = ('python', 'torch', 'cuda', 'cudnn', 'torchvision', 'numpy', 'timm', 'medmnist', 'sklearn')
if not ILK.exists():
    ILK.write_text(json.dumps(ortam, indent=1, ensure_ascii=False), encoding='utf-8'); print('ilk E1 ortamı kaydedildi')
else:
    ilk = json.loads(ILK.read_text(encoding='utf-8'))
    fark = {k: (ilk.get(k), ortam.get(k)) for k in CEKIRDEK if ilk.get(k) != ortam.get(k)}
    if fark:
        kq = DRIVE / 'runs' / 'kuyruklar'
        baslamis = [p.name for p in kq.glob('e1-*.baslat.json')] if kq.exists() else []
        assert not baslamis, (f'E1 kuyruğu {baslamis} başka çekirdek sürümle başladı; bu oturumun ortamı farklı: {fark}. '
                              'Seed\'ler/kollar iki ortama bölünemez → DUR, şefe bildir.')
        print('!!! UYARI: E1 ilk oturumundan farklı ortam (henüz başlamış E1 kuyruğu yok):', fark)
    else:
        print('ortam E1 ilk oturumuyla aynı (çekirdek sürümler)')
"""

C5 = V2.C5
C6 = V2.C6
C7 = V1.C6.replace("# 6) ÖNAĞIRLIKLAR", "# 7) ÖNAĞIRLIKLAR", 1)

C8 = r"""# 8) E1 HAZIRLIK: disk (kural b), kural (f) referansı, oturum koptuysa C1 verisi, e1-h10r ön koşulu (Durum B)
os.chdir(IS)
bos_gb = shutil.disk_usage(IS / 'data').free / 1e9
print(f'yerel disk boş: {bos_gb:.1f} GB (önbellek ≈ 26,2 GB; alt sınır {DISK_MIN_GB} GB)')
assert bos_gb >= DISK_MIN_GB, (f'DUR (H14 kural b): yerel diskte {bos_gb:.1f} GB < {DISK_MIN_GB} GB boş. Önbellek yazılmaz; '
                               'şefe bildir (alternatif yol yeni kart H14b).')
REF = IS / 'runs' / REF_RUN
assert (REF / 'metrics.json').exists() and (REF / 'tahmin_val.npy').exists() and (REF / 'config.json').exists(), \
    f'kural (f) referansı Drive\'da yok: yz/3-1/runs/{REF_RUN} — şefe bildir'
print('referans:', REF_RUN, '· ilk_adim_loss', json.loads((REF / 'metrics.json').read_text(encoding='utf-8'))['ilk_adim_loss'])

def e1_kontrol_durumu():
    # e1-h14 a1 kontrol kolunun metrics.json'u (yoksa None): kontrol_durum A/B/C
    son = None
    for d in sorted((IS / 'runs').glob('*_h14_c1_d4kapali_duman_s0')):
        if (d / '.done').exists() and (d / 'metrics.json').exists():
            c = json.loads((d / 'config.json').read_text(encoding='utf-8'))
            if str(c.get('kuyruk', '')).startswith('e1-h14'):
                m = json.loads((d / 'metrics.json').read_text(encoding='utf-8'))
                son = {'run': d.name, **{k: m.get(k) for k in ('kontrol_durum', 'kontrol_ilk_adim_loss_fark', 'kontrol_argmax_oran',
                                                               'kontrol_p_maks_fark', 'kontrol_surum_fark', 'disk_bos_gb_once',
                                                               'onbellek_d4_yazma_sn', 'onbellek_gorunum_tutarlilik_maks_fark')}}
    return son

if KUYRUK == 'e1-h10r':
    KD = e1_kontrol_durumu()
    assert KD is not None and KD['kontrol_durum'] == 'B', (f'e1-h10r YALNIZ kontrol kolu Durum B ise koşulur; şu an: {KD}. '
                                                           'Durum A → H10r gerekmez; Durum C → DUR. Şefe bildir.')
    print('Durum B doğrulandı → e1-h10r koşulabilir:', KD)
# Oturum koptuysa: devam kuyruğu a1'i geçtiyse C1 veri satırı (c_bolme_veri_c1) tekrar koşmaz → veri burada kurulur
# (aynı betik, aynı içerik SHA256 denetimi; ölçüm değildir). DermaMNIST-C arka plan indirmesini bekler.
for ad in KUYRUKLAR:
    aile_ = DY.aile(IS, ad)
    npz = IS / 'data' / 'dermamnist_c_resmipiksel_224.npz'
    if any(DY.baslamis(IS, x) for x in aile_) and DY.asama_gecti(IS, aile_, 1) and not npz.exists():
        hz = IS / 'e1_hazirlik' / f'c_bolme_{OTURUM}'
        hz.mkdir(parents=True, exist_ok=True)
        print('oturum devamı: C1 verisi bu oturumda yok → c_bolme_npz.py --kip c1 (kuyruk dışında, içerik SHA256 denetimli)')
        rc = kos(['python', 'scripts/c_bolme_npz.py', '--kip', 'c1', '--out', hz / 'metrics.json'], cwd=IS,
                 env={**os.environ, 'RUN_DIR': str(hz)})
        assert rc == 0 and npz.exists(), f'C1 verisi kurulamadı (rc={rc}) — DUR, şefe bildir'
        shutil.copytree(hz, DRIVE / 'durum' / hz.name, dirs_exist_ok=True)
print('E1 hazırlık tamam · KUYRUK', KUYRUK)
"""

C9 = r"""# 9) CANLI KALP ATIŞI: her 60 sn yz/3-1/durum/canli_<oturum>.json (bu sekmenin kuyruk ailelerinin aktif koşumları)
CANLI_KUYRUKLAR = list(KUYRUKLAR)   # Durum B'de e1-h10r bu sekmede başlarsa hücre 11 listeye ekler (iplik aynı listeyi okur)
CANLI_IPLIK, CANLI_DUR = DY.canli_baslat(IS, DURUM, KUYRUK, OTURUM, aralik=60, oturum_dosyasi=True, kuyruklar=CANLI_KUYRUKLAR,
                                         sekme=SEKME)
CANLI = DY.canli_dosyasi(DURUM, OTURUM)
for _ in range(90):
    if CANLI.exists(): break
    time.sleep(1)
print(CANLI.name, CANLI.read_text(encoding='utf-8')[:600] if CANLI.exists() else '(henüz yazılmadı — iş parçacığı çalışıyor; devam)')
"""

C10 = r"""# 10) KUYRUK — kademeli (a1 duman → a2 ön-uçuş → a3 tam + C-test); aşama kapıları otomatik; oturum koptuysa DEVAM.
#     Durum izleyici: kontrol kolu bitince durum/e1_durum_<oturum>.json yazılır ve aşağıda bir kez duyurulur.
import threading
CAKISMA = DY.es_zamanli_cakisma(IS, DURUM, KUYRUKLAR, OTURUM)
assert not CAKISMA, f'Bu kuyruk başka bir sekmede KOŞUYOR: {CAKISMA} — DUR (aynı kuyruğu iki sekme koşamaz)'
IZ_DUR = threading.Event()
def durum_izleyici():
    while not IZ_DUR.wait(30):
        kd = e1_kontrol_durumu()
        if kd:
            DY.atomik_json(DURUM / f'e1_durum_{OTURUM}.json', {**kd, 'zaman': DY.simdi(), 'oturum': OTURUM})
            print('\n' + '=' * 90 + f"\n[E1] KONTROL KOLU BİTTİ → DURUM {kd['kontrol_durum']} · {json.dumps(kd, ensure_ascii=False)}\n"
                  + {'A': '[E1] Durum A: arşiv h10_c1 K1 karşılaştırıcısıdır; H10r GEREKMEZ.',
                     'B': '[E1] Durum B: H10r ZORUNLU. e1-h14 bitince bu sekmede koşacak (H10R_AYNI_OTURUM) — ya da şimdi ikinci '
                          "sekmede KUYRUK='e1-h10r' ile başlatabilirsin (şefe sor).",
                     'C': '[E1] Durum C: DUR — kuyruk a1 kapısında duracak; şefe bildir.'}.get(kd['kontrol_durum'], '')
                  + '\n' + '=' * 90, flush=True)
            return
if KUYRUK == 'e1-h14':
    threading.Thread(target=durum_izleyici, name='e1-durum', daemon=True).start()
os.chdir(IS)
t0 = time.time()
GRUP_OZET = DY.grup_kos(IS, KUYRUK, durda_devam=DURDA_DEVAM, oturum=OTURUM)
IZ_DUR.set()
print(f'\nKUYRUK {KUYRUK} bitti · {(time.time() - t0) / 3600:.2f} sa')
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'aile'} for k, v in GRUP_OZET.items()}, indent=1, ensure_ascii=False))
"""

C11 = r"""# 11) DURUM B DALI (yalnız KUYRUK = e1-h14): kontrol kolu Durum B ise e1-h10r (H10 C1 tarifi aynen, aynı ortamda) koşulur.
KD = e1_kontrol_durumu()
print('kontrol kolu:', json.dumps(KD, ensure_ascii=False))
if KUYRUK != 'e1-h14':
    print('bu sekme e1-h10r koşturdu; dal atlandı')
elif KD is None:
    print('!!! kontrol kolu sonucu yok (a1 tamamlanmadı ya da çöktü) — şefe bildir')
elif KD['kontrol_durum'] == 'A':
    print('DURUM A → H10r GEREKMEZ. K1 karşılaştırıcısı arşiv h10_c1.')
elif KD['kontrol_durum'] == 'C':
    print('DURUM C → DUR (kod yolu hatası şüphesi). e1-h10r KOŞULMAZ. Şefe bildir.')
elif KD['kontrol_durum'] == 'B':
    plan = DY.devam_plani(IS, 'e1-h10r', uygula=False)
    diger = DY.es_zamanli_cakisma(IS, DURUM, ['e1-h10r'], OTURUM)
    if plan['durum'] in ('bitti', 'durdu'):
        print('e1-h10r zaten', plan['durum'], '(başka sekmede koşulmuş) — atlandı:', plan)
    elif diger:
        print('e1-h10r başka bir sekmede KOŞUYOR — bu sekmede başlatılmadı:', diger)
    elif H10R_AYNI_OTURUM:
        print('DURUM B → e1-h10r bu sekmede başlıyor (≈ 1,2 L4-sa)')
        CANLI_KUYRUKLAR.append('e1-h10r')   # kalp atışı artık e1-h10r'yi de kapsar (ikinci sekme aynı kuyruğu başlatamaz)
        DY.atomik_json(CANLI, DY.canli_durum(IS, KUYRUK, OTURUM, CANLI_KUYRUKLAR, SEKME))
        t1 = time.time()
        GRUP_OZET_R = DY.grup_kos(IS, 'e1-h10r', durda_devam=DURDA_DEVAM, oturum=OTURUM)
        print(f'e1-h10r bitti · {(time.time() - t1) / 3600:.2f} sa')
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'aile'} for k, v in GRUP_OZET_R.items()}, indent=1, ensure_ascii=False))
    else:
        print("DURUM B → H10R_AYNI_OTURUM = False: ikinci sekmede bu defteri KUYRUK = 'e1-h10r' ile aç ve Tümünü çalıştır.")
"""

C12 = r"""# 12) SON: damgalar + kuyruk listesi + a3 kapı raporları (bilgi) + E1 bütçesi + şefe gidecek Drive klasörleri.
#     Test sayısı GÖSTERİLMEZ (C-test analizi yalnız şefte, scripts/e1_h14_analiz.py, bir kez).
os.chdir(IS)
kos(['python', 'orkestra/kosum.py', '--kok', '.', 'durum'], cwd=IS)
kos(['python', 'orkestra/kuyruk.py', '--kok', '.', 'liste'], cwd=IS)
for ad in ('e1-h14', 'e1-h10r'):
    for x in DY.aile(IS, ad):
        if (IS / 'kuyruk' / f'{x}.jsonl').exists() and DY.kuyruk_damgasi(IS, x):
            kos(['python', 'orkestra/beklenti_kontrol.py', '--kok', '.', '--kuyruk', x, '--asama', '3'], cwd=IS)
kos(['python', 'orkestra/leaderboard.py', '--kok', '.', '--metrik', 'val_acc', '--yon', 'max', '--baseline', 'h10_c1',
     '--csv', f'sonuclar/leaderboard-colab-{KUYRUK}.csv', '--md', f'sonuclar/leaderboard-colab-{KUYRUK}.md'], cwd=IS)
# E1 bütçesi (kural h): e1-* kuyruklarının koşum süreleri (L4-sa)
e1_sn = 0.0
for d in (IS / 'runs').glob('*/'):
    if (d / 'config.json').exists() and (d / '.done').exists():
        c = json.loads((d / 'config.json').read_text(encoding='utf-8'))
        if str(c.get('kuyruk', '')).startswith('e1-'):
            e1_sn += float(json.loads((d / '.done').read_text(encoding='utf-8')).get('sure_sn') or 0)
print(f'E1 birleşik koşum süresi: {e1_sn / 3600:.2f} L4-sa (kart tavanı 5,0)')
ozet = {'oturum': OTURUM, 'kuyruk': KUYRUK, 'kontrol': e1_kontrol_durumu(), 'e1_l4_sa': round(e1_sn / 3600, 3),
        'zaman': DY.simdi(), 'torch_sabitleme': PIN, 'ekim_farki': EKIM_FARK}
DY.atomik_json(DURUM / f'e1_bitti_{OTURUM}.json', ozet)
print('DermaMNIST-C arka plan:', VERI_C_JSON.read_text(encoding='utf-8')[:300] if VERI_C_JSON.exists() else 'henüz yok')
CANLI_DUR.set(); time.sleep(1)
son = DY.canli_durum(IS, KUYRUK, OTURUM, CANLI_KUYRUKLAR, SEKME); son['bitti'] = True
DY.atomik_json(CANLI, son)
print(f'''BİTTİ (sekme {SEKME}, {KUYRUK}). Sonuçlar Drive'da kalır; zip/indirme GEREKMEZ. Şefe giden klasörler (Drive for desktop eşitler):
  yz/3-1/runs/*_h14_c1*  ve (Durum B) *_h10r_c1*  ·  yz/3-1/runs/kuyruklar/e1-*  ·  yz/3-1/sonuclar/beklenti-e1-*
  yz/3-1/env/*{OTURUM}*  ·  yz/3-1/durum/e1_*{OTURUM}*.json
Şefe "E1 bitti" demen yeterli. Bu sekmeyi kapatabilirsin.''')
"""


def defter() -> dict:
    hucreler = [V1.hucre("markdown", MD0)] + [V1.hucre("code", c) for c in (C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12)]
    return {"nbformat": 4, "nbformat_minor": 0,
            "metadata": {"colab": {"provenance": [], "gpuType": "L4", "machine_shape": "hm"},
                         "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                         "accelerator": "GPU"},
            "cells": hucreler}


def main():
    nb = defter()
    for i, h in enumerate(nb["cells"]):  # sözdizimi denetimi (çalıştırılmaz)
        if h["cell_type"] == "code":
            compile("".join(h["source"]), f"e1-hücre{i}", "exec")
    CIKTI.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"yazıldı: {CIKTI} ({len(nb['cells'])} hücre; kod hücreleri derlendi)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
