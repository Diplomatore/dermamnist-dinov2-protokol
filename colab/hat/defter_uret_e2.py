#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""defter_uret_e2.py — E2 (H15, BCN20000 dış merkez) Colab defteri `colab/e2-h15.ipynb`'yi üretir (KULLANICI ONAYI H15, 2026-11-07).

Hat hücreleri yeniden kullanılır (defter_uret.py v2: kurulum/paket/veri; defter_uret_v1.py: donanım + önağırlık/öz-sınama;
defter_uret_e1.py: torch sabitleme hücresi). E2'ye özgü olanlar:
  hücre 1  PARAMETRELER — DRIVE_YZ (kişisel yol), KUYRUK = e2-degerlendirme (tek kuyruk), torch sabitleme hedefi (kart: E1 ile aynı).
  hücre 4  ORTAM: torch 2.11.0+cu130 sabitleme (E1 hücresi) + Ekim config.json ve E1 ilk oturumuyla sürüm farkı (timm, numpy,
           Pillow, cuDNN yazılır). Sürüklenme denetimi E2'nin ilk oturumuna göre (env/ortam-colab-e2-ilk.json).
  hücre 8  E2 HAZIRLIK: ETİKETSİZ BCN girdisi Drive `yz/3-1/veri_dis/bcn20000/e2_girdi/` → /content/veri (dosya SHA256 =
           kuyruk/e2_bcn_beklenen.json) → `bcn_veri.py ac` (npz → npy, içerik SHA256); disk; oturum koptuysa C1 verisi.
  hücre 10 KUYRUK (kademeli a1 → a2 → a3; a3 ilk satırı KİLİDİ açar).
  hücre 11 SON: damgalar, a3 kapı raporu, E2 bütçesi, Drive'da şefe gidecek klasörler. BCN SAYISI GÖSTERİLMEZ (etiket Colab'da yok;
           analiz yalnız şefte, yerelde, bir kez).
Kullanım: <uv-python> colab/hat/defter_uret_e2.py → colab/e2-h15.ipynb (drive_paketle.py Drive'a yz/e2-h15.ipynb olarak kopyalar)."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HAT = Path(__file__).resolve().parent
sys.path.insert(0, str(HAT))
import defter_uret as V2  # noqa: E402
import defter_uret_e1 as E1  # noqa: E402
import defter_uret_v1 as V1  # noqa: E402

WS = HAT.parents[1]
CIKTI = WS / "colab" / "e2-h15.ipynb"

MD0 = r"""# 3.1 DermaMNIST — E2 / H15 Colab defteri: BCN20000 dış merkez, yeniden eğitimsiz, KİLİTLİ (KULLANICI ONAYI H15, 2026-11-07)

**Ne yapar:** Arşivdeki 11 eğitilmiş kolu (55 kol-seed) yeniden eğitmeden yükler, iç val özdeşliğini sınar ve BCN20000
(12.413 görüntü, iki ön-işleme) için olasılık dosyası yazar. Kuyruk kademeli: a1 veri kapısı + duman → a2 ön-uçuş (val
özdeşliği, BCN yok) → a3 KİLİT + BCN çıkarımı (**her kol-seed bir kez**). Aşama kapıları otomatiktir.

**Kurallar (hipotezler/H15.md):**
- BCN **etiketleri Colab'a hiç gelmez**; model girdisi etiketsizdir. Bu defter BCN doğruluğu **göstermez** (analiz şefte, bir kez).
- İç test kümeleri (resmî test, C-test) **açılmaz**. C1 kolları yalnız C-val'de doğrulanır.
- Eşik/sıcaklık/önsel ayarı yok; BCN üzerinde hiçbir seçim yok. Bütçe tavanı 3,0 L4-sa (aşılırsa satırlar kendiliğinden başlamaz).

**Senin yapacağın:** ilk hücredeki `DRIVE_YZ` yolunu kontrol et → *Çalışma zamanı → Çalışma zamanı türünü değiştir →
**L4 GPU*** → *Tümünü çalıştır* → sekmeyi açık tut. Ayrıntı: `colab/E2-OKU-BENI.md`.
"""

C1 = r"""# 1) PARAMETRELER — kişiye özel yol (yalnız DRIVE_YZ satırını düzenleyebilirsin)
DRIVE_YZ = '<DRIVE_KOK>'   # Drive'daki 'yz' klasörü (Drive for desktop'ta: <DRIVE_KOK>). Farklıysa düzelt.
KUYRUK = 'e2-degerlendirme'   # tek kuyruk (a1 → a2 → a3); DEĞİŞTİRME
DURDA_DEVAM = False           # bir aşama kapısı DURURSA devam edilmez (şef teşhis eder)
TORCH_PIN = '2.11.0+cu130'           # kart "Ortam": E1 ile aynı sabitleme
TORCHVISION_PIN = '0.26.0+cu130'
PIN_INDEX = 'https://download.pytorch.org/whl/cu130'
EKIM_ORTAM = {'torch': '2.11.0+cu130', 'cuda': '13.0', 'cudnn': 92700, 'timm': '1.0.30', 'numpy': '2.1.3'}   # Ekim kolları config.json
DISK_MIN_GB = 20              # BCN girdisi (npz ≈ 2,5 GB + açık npy ≈ 3,7 GB) + C1 verisi + öznitelik önbellekleri
SEKME = 'E2'
assert KUYRUK == 'e2-degerlendirme', f'KUYRUK yalnız e2-degerlendirme olabilir: {KUYRUK!r}'
"""

C2 = V2.C2.replace("YZ = pathlib.Path('<DRIVE_KOK>')", "YZ = pathlib.Path(DRIVE_YZ)   # hücre 1 parametresi", 1)
assert C2 != V2.C2, "defter_uret.C2 Drive yolu satırı bulunamadı"

C3 = V1.C2.replace("# 2) DONANIM", "# 3) DONANIM", 1)


def _c4() -> str:
    c = E1.C4
    deg = [("# 4) ORTAM (E1): torch'u Ekim sürümüne SABİTLEMEYİ DENE (H14 \"Ortam\": Durum A olasılığı) → olmazsa mevcut sürümle devam\n"
            "#    (Durum B yolu; sürümler kaydedilir).",
            "# 4) ORTAM (E2): torch'u 2.11.0+cu130'a SABİTLEMEYİ DENE (H15 \"Ortam\"; E1 ile aynı) → olmazsa mevcut sürümle devam\n"
            "#    (val özdeşliği Durum B'ye düşebilir; sürümler kaydedilir)."),
           ("print('Ekim (H10 C1) ortamına göre fark:', EKIM_FARK or 'YOK → Durum A beklenir',\n"
            "      '' if not EKIM_FARK else '→ Durum B olası (kontrol kolu karar verir; e1-h10r gerekebilir)')",
            "print('Ekim kollarının ortamına göre fark (timm/numpy/cuDNN dahil):', EKIM_FARK or 'YOK → Durum A beklenir',\n"
            "      '' if not EKIM_FARK else '→ özdeşlik Durum B olabilir (kart K-c; kabul ölçütü argmax ≥ 0,995 ∧ maks|Δp| ≤ 1e-3)')\n"
            "_e1 = DRIVE / 'env' / 'ortam-colab-e1-ilk.json'\n"
            "if _e1.exists():\n"
            "    _e1d = json.loads(_e1.read_text(encoding='utf-8'))\n"
            "    E1_FARK = {k: [_e1d.get(k), ortam.get(k)] for k in ('torch', 'cuda', 'cudnn', 'timm', 'numpy', 'PIL') if _e1d.get(k) != ortam.get(k)}\n"
            "    print('E1 (h14_c1) ortamına göre fark (Pillow dahil):', E1_FARK or 'YOK')\n"
            "else:\n"
            "    E1_FARK = None"),
           ("'kuyruk': KUYRUK, 'kuyruklar': KUYRUKLAR, 'nvidia_smi': gpu, 'torch_sabitleme': PIN, 'ekim_farki': EKIM_FARK,",
            "'kuyruk': KUYRUK, 'kuyruklar': KUYRUKLAR, 'nvidia_smi': gpu, 'torch_sabitleme': PIN, 'ekim_farki': EKIM_FARK, 'e1_farki': E1_FARK,"),
           ("# Sürüklenme: E1'in İLK oturumuna göre (env/ortam-colab-e1-ilk.json). Başlamış bir E1 kuyruğunun devamında ya da Durum B'de\n"
            "# H10r'nin H14'ten farklı ortamda koşması → DUR (karşılaştırma aynı ortamda olmalı).\n"
            "ILK = ENV / 'ortam-colab-e1-ilk.json'",
            "# Sürüklenme: E2'nin İLK oturumuna göre (env/ortam-colab-e2-ilk.json). Başlamış e2 kuyruğunun devamında çekirdek sürüm\n"
            "# değiştiyse → DUR (55 kol-seed iki ortama bölünmez).\n"
            "ILK = ENV / 'ortam-colab-e2-ilk.json'"),
           ("ILK.write_text(json.dumps(ortam, indent=1, ensure_ascii=False), encoding='utf-8'); print('ilk E1 ortamı kaydedildi')",
            "ILK.write_text(json.dumps(ortam, indent=1, ensure_ascii=False), encoding='utf-8'); print('ilk E2 ortamı kaydedildi')"),
           ("baslamis = [p.name for p in kq.glob('e1-*.baslat.json')] if kq.exists() else []",
            "baslamis = [p.name for p in kq.glob('e2-*.baslat.json')] if kq.exists() else []"),
           ("assert not baslamis, (f'E1 kuyruğu {baslamis} başka çekirdek sürümle başladı; bu oturumun ortamı farklı: {fark}. '",
            "assert not baslamis, (f'E2 kuyruğu {baslamis} başka çekirdek sürümle başladı; bu oturumun ortamı farklı: {fark}. '"),
           ("print('!!! UYARI: E1 ilk oturumundan farklı ortam (henüz başlamış E1 kuyruğu yok):', fark)",
            "print('!!! UYARI: E2 ilk oturumundan farklı ortam (henüz başlamış E2 kuyruğu yok):', fark)"),
           ("print('ortam E1 ilk oturumuyla aynı (çekirdek sürümler)')", "print('ortam E2 ilk oturumuyla aynı (çekirdek sürümler)')")]
    for a, b in deg:
        assert c.count(a) == 1, a[:80]
        c = c.replace(a, b)
    return c


C4 = _c4()
C5 = V2.C5
C6 = V2.C6
C7 = V1.C6.replace("# 6) ÖNAĞIRLIKLAR", "# 7) ÖNAĞIRLIKLAR", 1)

C8 = r"""# 8) E2 HAZIRLIK: ETİKETSİZ BCN girdisi Drive → /content/veri (dosya SHA256) → açma (npz → npy, içerik SHA256);
#    disk; oturum koptuysa C1 verisi. BCN etiketi (tanı/lezyon) bu oturuma HİÇ gelmez.
os.chdir(IS)
B_E2 = json.loads((IS / 'kuyruk' / 'e2_bcn_beklenen.json').read_text(encoding='utf-8'))   # hücre 5'te PAKET SHA256'sıyla doğrulandı
VD = DRIVE / 'veri_dis' / 'bcn20000' / 'e2_girdi'
YD = VERI / 'bcn20000' / 'e2_girdi'
YD.mkdir(parents=True, exist_ok=True)
bag = IS / 'data' / 'bcn20000'
if bag.is_symlink() or bag.is_file():
    bag.unlink()
elif bag.exists():
    shutil.rmtree(bag)
bag.symlink_to(VERI / 'bcn20000', target_is_directory=True)
for ad, bilgi in B_E2['dosyalar'].items():
    k, h = VD / ad, YD / ad
    assert k.exists(), (f'{ad} Drive\'da yok (yz/3-1/veri_dis/bcn20000/e2_girdi/). Şef colab/hat/e2_veri_kopya.py koşmalı ya da '
                        'Drive yüklemesi bitmemiş → birkaç dakika bekleyip defteri yeniden çalıştır.')
    if not (h.exists() and DY.sha256(h) == bilgi['sha256']):
        t1 = time.time(); shutil.copyfile(k, h)
        print(f'{ad} kopyalandı ({h.stat().st_size / 1e9:.2f} GB, {time.time() - t1:.0f} sn)')
    assert DY.sha256(h) == bilgi['sha256'], f'{ad} SHA256 beklenenle aynı değil (Drive eşitlemesi yarım?) — DUR, şefe bildir'
print('BCN girdi dosyaları SHA256 doğrulandı:', list(B_E2['dosyalar']))
rc = kos(['python', 'varyantlar/e2/bcn_veri.py', 'ac'], cwd=IS)
assert rc == 0, f'BCN girdisi açılamadı / içerik SHA256 tutmadı (rc={rc}) — DUR (K-b), şefe bildir'
bos_gb = shutil.disk_usage(IS / 'data').free / 1e9
print(f'yerel disk boş: {bos_gb:.1f} GB (alt sınır {DISK_MIN_GB} GB)')
assert bos_gb >= DISK_MIN_GB, f'DUR: yerel diskte {bos_gb:.1f} GB < {DISK_MIN_GB} GB boş — şefe bildir'
# Oturum koptuysa: devam kuyruğu a1'i geçtiyse C1 veri satırı yeniden koşmaz → C1 verisi burada kurulur (E1 ile aynı yol)
npz_c1 = IS / 'data' / 'dermamnist_c_resmipiksel_224.npz'
aile_ = DY.aile(IS, KUYRUK)
if any(DY.baslamis(IS, x) for x in aile_) and DY.asama_gecti(IS, aile_, 1) and not npz_c1.exists():
    hz = IS / 'e2_hazirlik' / f'c_bolme_{OTURUM}'
    hz.mkdir(parents=True, exist_ok=True)
    print('oturum devamı: C1 verisi bu oturumda yok → c_bolme_npz.py --kip c1 (kuyruk dışında, içerik SHA256 denetimli)')
    rc = kos(['python', 'scripts/c_bolme_npz.py', '--kip', 'c1', '--out', hz / 'metrics.json'], cwd=IS,
             env={**os.environ, 'RUN_DIR': str(hz)})
    assert rc == 0 and npz_c1.exists(), f'C1 verisi kurulamadı (rc={rc}) — DUR, şefe bildir'
    shutil.copytree(hz, DRIVE / 'durum' / hz.name, dirs_exist_ok=True)
KILIT_DRIVE = IS / 'sonuclar' / 'e2_bcn.KILIT'
print('E2 hazırlık tamam · KİLİT', 'AÇIK (önceki oturumda a3 başladı)' if KILIT_DRIVE.exists() else 'henüz yok (a3 ilk satırı açar)')
"""

C9 = r"""# 9) CANLI KALP ATIŞI: her 60 sn yz/3-1/durum/canli_<oturum>.json (aktif koşum, ilerleme, GPU bellek)
CANLI_IPLIK, CANLI_DUR = DY.canli_baslat(IS, DURUM, KUYRUK, OTURUM, aralik=60, oturum_dosyasi=True, kuyruklar=KUYRUKLAR,
                                         sekme=SEKME)
CANLI = DY.canli_dosyasi(DURUM, OTURUM)
for _ in range(90):
    if CANLI.exists(): break
    time.sleep(1)
print(CANLI.name, CANLI.read_text(encoding='utf-8')[:600] if CANLI.exists() else '(henüz yazılmadı — iş parçacığı çalışıyor; devam)')
"""

C10 = r"""# 10) KUYRUK — kademeli: a1 veri kapısı + C1 verisi + H10 dumanı → a2 ön-uçuş (55; val özdeşliği) → a3 KİLİT + BCN (55).
#     Aşama kapıları otomatik; oturum koptuysa DEVAM (bitmiş satırlar atlanır; aynı kol-seed için ikinci BCN tahmini üretilmez).
CAKISMA = DY.es_zamanli_cakisma(IS, DURUM, KUYRUKLAR, OTURUM)
assert not CAKISMA, f'Bu kuyruk başka bir sekmede KOŞUYOR: {CAKISMA} — DUR (aynı kuyruğu iki sekme koşamaz)'
os.chdir(IS)
t0 = time.time()
GRUP_OZET = DY.grup_kos(IS, KUYRUK, durda_devam=DURDA_DEVAM, oturum=OTURUM)
print(f'\nKUYRUK {KUYRUK} bitti · {(time.time() - t0) / 3600:.2f} sa')
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'aile'} for k, v in GRUP_OZET.items()}, indent=1, ensure_ascii=False))
"""

C11 = r"""# 11) SON: damgalar + kuyruk listesi + a3 kapı raporu (bilgi) + E2 bütçesi + şefe gidecek Drive klasörleri.
#     BCN doğruluğu GÖSTERİLMEZ ve hesaplanamaz (etiket Colab'da yok). Analiz yalnız şefte: scripts/e2_bcn_analiz.py, bir kez.
os.chdir(IS)
kos(['python', 'orkestra/kosum.py', '--kok', '.', 'durum'], cwd=IS)
kos(['python', 'orkestra/kuyruk.py', '--kok', '.', 'liste'], cwd=IS)
for x in DY.aile(IS, KUYRUK):
    if (IS / 'kuyruk' / f'{x}.jsonl').exists() and DY.kuyruk_damgasi(IS, x):
        kos(['python', 'orkestra/beklenti_kontrol.py', '--kok', '.', '--kuyruk', x, '--asama', '3'], cwd=IS)
e2_sn = 0.0
durum_say = {}
for d in (IS / 'runs').glob('*/'):
    if (d / 'config.json').exists() and (d / '.done').exists():
        c = json.loads((d / 'config.json').read_text(encoding='utf-8'))
        if str(c.get('kuyruk', '')).startswith(KUYRUK):
            e2_sn += float(json.loads((d / '.done').read_text(encoding='utf-8')).get('sure_sn') or 0)
            if str(c.get('varyant', '')).endswith('-bcn') and (d / 'metrics.json').exists():
                m = json.loads((d / 'metrics.json').read_text(encoding='utf-8'))
                dk = {1: 'A', 2: 'B', 3: 'C'}.get(m.get('ozdeslik_durum_kod'), '?')
                durum_say[dk] = durum_say.get(dk, 0) + 1
print(f'E2 koşum süresi: {e2_sn / 3600:.2f} L4-sa (kart tavanı 3,0) · a3 özdeşlik durumları {durum_say}')
ozet = {'oturum': OTURUM, 'kuyruk': KUYRUK, 'e2_l4_sa': round(e2_sn / 3600, 3), 'a3_durum': durum_say, 'zaman': DY.simdi(),
        'torch_sabitleme': PIN, 'ekim_farki': EKIM_FARK, 'kilit_var': (IS / 'sonuclar' / 'e2_bcn.KILIT').exists()}
DY.atomik_json(DURUM / f'e2_bitti_{OTURUM}.json', ozet)
CANLI_DUR.set(); time.sleep(1)
son = DY.canli_durum(IS, KUYRUK, OTURUM, KUYRUKLAR, SEKME); son['bitti'] = True
DY.atomik_json(CANLI, son)
print(f'''BİTTİ (sekme {SEKME}, {KUYRUK}). Sonuçlar Drive'da kalır; zip/indirme GEREKMEZ. Şefe giden klasörler (Drive for desktop eşitler):
  yz/3-1/runs/*_e2_*  ·  yz/3-1/runs/kuyruklar/e2-*  ·  yz/3-1/sonuclar/e2_bcn.KILIT  ·  yz/3-1/sonuclar/beklenti-e2-*
  yz/3-1/env/*{OTURUM}*  ·  yz/3-1/durum/e2_*{OTURUM}*.json
Şefe "E2 bitti" demen yeterli. Bu sekmeyi kapatabilirsin.''')
"""


def defter() -> dict:
    hucreler = [V1.hucre("markdown", MD0)] + [V1.hucre("code", c) for c in (C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11)]
    return {"nbformat": 4, "nbformat_minor": 0,
            "metadata": {"colab": {"provenance": [], "gpuType": "L4", "machine_shape": "hm"},
                         "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                         "accelerator": "GPU"},
            "cells": hucreler}


def main():
    nb = defter()
    for i, h in enumerate(nb["cells"]):  # sözdizimi denetimi (çalıştırılmaz)
        if h["cell_type"] == "code":
            compile("".join(h["source"]), f"e2-hücre{i}", "exec")
    CIKTI.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"yazıldı: {CIKTI} ({len(nb['cells'])} hücre; kod hücreleri derlendi)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
