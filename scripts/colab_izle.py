#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
colab_izle.py — YEREL İZLEYİCİ (Kanal 2 karşılığı, Colab hattı): Google Drive for desktop ile eşitlenen
<DRIVE_KOK>\\3-1 klasöründeki damgaları okuyup OLAY SATIRLARI basar. torch gerekmez (uv Python 3.11 yeter).

İki kip:
  --grup <grup>                 (v1) kuyruk/gruplar.json'daki grup tek sekmede SIRAYLA koşar; bir kuyruk DURURSA grup durur
                                → izleyici çıkar. Grup adı gruplarda yoksa tek kuyruk sayılır.
  --kuyruklar a,b,c             (v2, PARALEL sekmeler) her kuyruk ayrı sekmede/oturumda; her biri bağımsız biter/durur.
                                Bir kuyruğun DURMASI diğerlerini izlemeyi kesmez; HEPSİ bitince/durunca çıkar.

Olaylar (her biri tek satır; Monitor bildirimine uygun):
  BAŞLADI <run_id> …              koşum klasörü (config.json) belirdi, damga yok
  BİTTİ <run_id> rc=… val_acc=…   .done belirdi (rc≠0 ise neden: .erken_durdurma.json / erteleme.json)
  AŞAMA-GEÇTİ / AŞAMA-BAŞLADI / KUYRUK-BAŞLADI / KUYRUK-KOPTU / YETİM-DAMGA
                                  runs/BILDIRIM.log (v1) ve runs/BILDIRIM_<oturum>.log (v2, oturum başına) yeni satırları
  ASILI <canlı dosyası> …         (a) kuyruğu kapsayan en yeni canlı dosyanın kalp atışı --asili-dk'dan bayat
                                  (o sekme/oturum kopmuş olabilir), (b) aktif koşumun Colab-tarafı kalp atışı > 600 s bayat
  CANLI-YOK <kuyruk> …            (--kuyruklar) izleme başlayalı --asili-dk geçti, bu kuyruğun sekmesinden hiç kalp atışı yok
  HATA-LOG <dosya> …              durum/hata_*.log yeni belirdi (v2 adı hata_<oturum>_<zaman>.log; ilk/son satırla)
  KUYRUK-DURDU <ad> asama=k …     KUYRUK.done durduruldu_asama ≠ null (beklenti kapısı)
  KUYRUK-BİTTİ <ad> …             ailenin son kuyruğunda KUYRUK.done
  GRUP-BİTTİ / GRUP-DURDU         (--grup) grubun tüm kuyrukları bitti / biri durdu → ÇIKAR
  TÜMÜ-BİTTİ …                    (--kuyruklar) her kuyruk bitti ya da durdu → ÇIKAR
Canlı dosyalar: durum/canli.json (v1; 'grup' alanı) ve durum/canli_<oturum>.json (v2; 'kuyruklar' + 'sekme' alanı).
Bir kuyruk için, onu kapsayan canlı dosyalardan kalp_epoch'u EN YENİ olan esas alınır (sekme yeniden açılınca yeni
oturum dosyası eskisini geçer). Aktif koşum listesi kuyruğun ailesine göre süzülür.
PID denetimi YAPILMAZ (Colab PID'leri yerelde anlamsız). Drive eşitleme gecikmesi nedeniyle ASILI eşiği ≥ 15 dk.
Zaman: canlı dosyadaki kalp_epoch (UNIX zamanı, saat dilimi bağımsız) yerel time.time() ile kıyaslanır.

Kullanım:
  <uv-python> scripts/colab_izle.py --grup kc1-tekrar [--drive "<DRIVE_KOK>/3-1"] [--aralik 60] [--asili-dk 20]
  <uv-python> scripts/colab_izle.py --kuyruklar kc2-tur4-h10,kc2-tur4-h11,kc2-tur4-h12 [--aralik 60] [--asili-dk 20]
  <uv-python> scripts/colab_izle.py --kendi-testi
Çıkış: 0 hepsi bitti (başarısız koşum yok) · 1 bitti, başarısız koşum var · 3 en az bir kuyruk aşama kapısında DURDU
"""
import argparse
import json
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:  # noqa: BLE001
    pass

VARSAYILAN_DRIVE = "<DRIVE_KOK>/3-1"
BILDIRIM_OLAYLARI = ("AŞAMA-GEÇTİ", "AŞAMA-BAŞLADI", "KUYRUK-BAŞLADI", "KUYRUK-KOPTU", "YETİM-DAMGA")


def oku_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — eşitleme sırasında yarım dosya olabilir
        return None


def aile_uyesi(kosum_kuyrugu, kuyruklar) -> bool:
    if not kosum_kuyrugu:
        return False
    for q in kuyruklar:
        if kosum_kuyrugu == q or (kosum_kuyrugu.startswith(q + "-devam") and kosum_kuyrugu[len(q) + 6:].isdigit()):
            return True
    return False


class Izleyici:
    def __init__(self, kok: Path, grup: str = "kc1-tekrar", asili_dk: float = 20.0, saat=time.time, kuyruklar=None):
        self.kok, self.asili_sn, self.saat = Path(kok), asili_dk * 60, saat
        self.coklu = bool(kuyruklar)
        self.secili = list(kuyruklar) if kuyruklar else None
        self.grup = ",".join(self.secili) if self.coklu else grup
        self.gorulen_run = {}      # run_id → 'BASLADI' | 'BITTI'
        self.gorulen_hata = set()
        self.bildirim_satir = {}   # BILDIRIM*.log adı → okunan satır sayısı
        self.son_asili = {}        # birim → son ASILI bildirimi
        self.son_kosum_asili = {}
        self.son_canli_yok = {}
        self.bitti_bildirilen = set()
        self.birim_sonuc = {}      # (--kuyruklar) kuyruk → 0 | 1 | 3
        self.baslangic = saat()

    # ── yardımcılar ──
    def gruplar(self) -> dict:
        return (oku_json(self.kok / "kuyruk" / "gruplar.json") or {}).get("gruplar") or {}

    def kuyruklar(self) -> list:
        if self.coklu:
            return list(self.secili)
        return self.gruplar().get(self.grup, [self.grup])

    def birimler(self) -> list:
        """Bağımsız biten birimler: --kuyruklar → her kuyruk; --grup → grubun tamamı tek birim."""
        return [(q, [q]) for q in self.secili] if self.coklu else [(self.grup, self.kuyruklar())]

    def aile(self, ad: str) -> list:
        out = [ad]
        d = []
        for p in (self.kok / "kuyruk").glob(f"{ad}-devam*.jsonl"):
            ek = p.stem[len(ad) + len("-devam"):]
            if ek.isdigit():
                d.append((int(ek), p.stem))
        return out + [s for _, s in sorted(d)]

    def tum_adlar(self) -> set:
        return {x for ad in self.kuyruklar() for x in self.aile(ad)}

    def damga(self, ad: str):
        return oku_json(self.kok / "runs" / "kuyruklar" / f"{ad}.KUYRUK.done")

    def canli_dosyalari(self) -> list:
        dz = self.kok / "durum"
        out = []
        for p in sorted(dz.glob("canli*.json")) if dz.exists() else []:
            cj = oku_json(p)
            if isinstance(cj, dict):
                out.append((p, cj))
        return out

    def canli_kapsam(self, cj: dict, G: dict):
        """None → kapsam bilgisi yok (eski biçim; her şeye uyar)."""
        if cj.get("kuyruklar"):
            return set(cj["kuyruklar"])
        g = cj.get("grup")
        return set(G.get(g, [g])) if g else None

    # ── tek tarama ──
    def tara(self, ilk: bool = False) -> tuple[list, int | None]:
        olaylar, adlar = [], self.tum_adlar()
        runs = self.kok / "runs"
        # 1) koşumlar
        if runs.exists():
            for d in sorted(runs.iterdir()):
                if not d.is_dir() or d.name == "kuyruklar":
                    continue
                c = oku_json(d / "config.json")
                if not c or c.get("kuyruk") not in adlar:
                    continue
                done = oku_json(d / ".done") if (d / ".done").exists() else None
                onceki = self.gorulen_run.get(d.name)
                if done is not None and onceki != "BITTI":
                    if not ilk:
                        m = done.get("metrics") or {}
                        ek = ""
                        if (d / ".erken_durdurma.json").exists():
                            ek = " ERKEN-DURDURULDU: " + str((oku_json(d / ".erken_durdurma.json") or {}).get("neden", ""))
                        if (d / "erteleme.json").exists():
                            e = oku_json(d / "erteleme.json") or {}
                            ek += f" ERTELEME {e.get('karar')} sayaç {e.get('sayac')} / eşik {e.get('esik')}"
                        va = m.get("val_acc")
                        olaylar.append(f"BİTTİ {d.name} rc={done.get('rc')} {done.get('sure_sn', '?')}s "
                                       f"a{c.get('asama')} {c.get('varyant')} s{c.get('seed')}"
                                       f"{f' val_acc={va:.4f}' if isinstance(va, (int, float)) else ' METRİK YOK' if not m else ''}{ek}")
                    self.gorulen_run[d.name] = "BITTI"
                elif done is None and onceki is None:
                    if not ilk:
                        olaylar.append(f"BAŞLADI {d.name} a{c.get('asama')} {c.get('varyant')} s{c.get('seed')} kuyruk={c.get('kuyruk')}")
                    self.gorulen_run[d.name] = "BASLADI"
        # 2) BILDIRIM.log (v1) + BILDIRIM_<oturum>.log (v2) yeni satırlar (izlenen ailelere ait kuyruk olayları)
        for bl in sorted(runs.glob("BILDIRIM*.log")) if runs.exists() else []:
            try:
                satirlar = bl.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            n0 = self.bildirim_satir.get(bl.name, 0)
            yeni = satirlar[n0:] if len(satirlar) >= n0 else satirlar
            self.bildirim_satir[bl.name] = len(satirlar)
            if not ilk:
                for s in yeni:
                    p = s.split()
                    if len(p) >= 3 and p[1] in BILDIRIM_OLAYLARI and (p[2] in adlar or p[1] == "YETİM-DAMGA"):
                        olaylar.append(" ".join(p[1:]))
        # 3) hata günlükleri
        dz = self.kok / "durum"
        if dz.exists():
            for h in sorted(dz.glob("hata_*.log")):
                if h.name in self.gorulen_hata:
                    continue
                self.gorulen_hata.add(h.name)
                if not ilk:
                    try:
                        t = h.read_text(encoding="utf-8", errors="replace").strip().splitlines()
                    except OSError:
                        t = []
                    son = next((x for x in reversed(t) if x.strip()), "")
                    olaylar.append(f"HATA-LOG {h} | {t[0] if t else ''} | son: {son[:200]}")
        # 4) kuyruk damgaları → bitiş
        if self.coklu:
            rc = self._bitis_coklu(olaylar)
        else:
            rc = self._bitis_grup(olaylar)
        if rc is not None:
            return olaylar, rc
        # 5) asılı denetimi (bitmemiş birimler)
        if not ilk:
            G = self.gruplar()
            for ad, kq in self.birimler():
                if self.coklu and ad in self.birim_sonuc:
                    continue
                olaylar += self.asili(ad, kq, G)
        return olaylar, None

    def _bitis_grup(self, olaylar: list):
        toplam_basarisiz, hepsi_bitti = 0, True
        for ad in self.kuyruklar():
            aile = self.aile(ad)
            durdu = next(((x, self.damga(x)) for x in aile if (self.damga(x) or {}).get("durduruldu_asama") is not None), None)
            if durdu:
                x, j = durdu
                if x not in self.bitti_bildirilen:
                    self.bitti_bildirilen.add(x)
                    olaylar.append(self._durdu_satiri(x, j))
                olaylar.append(f"GRUP-DURDU {self.grup} ({x})")
                return 3
            j = self.damga(aile[-1])
            if j is None:
                hepsi_bitti = False
                break
            if aile[-1] not in self.bitti_bildirilen:
                self.bitti_bildirilen.add(aile[-1])
                ok, top, bas = self.aile_ozet(aile)
                olaylar.append(f"KUYRUK-BİTTİ {ad} (aile {len(aile)} kuyruk) son={aile[-1]} aşama-3 başarılı {ok}/{top}, "
                               f"başarısız koşum {bas}")
            toplam_basarisiz += self.aile_ozet(aile)[2]
        if hepsi_bitti:
            olaylar.append(f"GRUP-BİTTİ {self.grup} başarısız koşum {toplam_basarisiz} → kosum.py durum + leaderboard + kapı")
            return 0 if toplam_basarisiz == 0 else 1
        return None

    def _bitis_coklu(self, olaylar: list):
        for q in self.secili:
            if q in self.birim_sonuc:
                continue
            aile = self.aile(q)
            durdu = next(((x, self.damga(x)) for x in aile if (self.damga(x) or {}).get("durduruldu_asama") is not None), None)
            if durdu:
                x, j = durdu
                olaylar.append(self._durdu_satiri(x, j) + " (diğer kuyruklar izlenmeye devam)")
                self.birim_sonuc[q] = 3
                continue
            if self.damga(aile[-1]) is None:
                continue
            ok, top, bas = self.aile_ozet(aile)
            olaylar.append(f"KUYRUK-BİTTİ {q} (aile {len(aile)} kuyruk) son={aile[-1]} aşama-3 başarılı {ok}/{top}, "
                           f"başarısız koşum {bas}")
            self.birim_sonuc[q] = 0 if bas == 0 else 1
        if len(self.birim_sonuc) == len(self.secili):
            v = self.birim_sonuc
            durum = " ".join(f"{q}={'DURDU' if v[q] == 3 else ('başarısız-var' if v[q] == 1 else 'tamam')}" for q in self.secili)
            olaylar.append(f"TÜMÜ-BİTTİ {durum} → kosum.py durum + leaderboard + kapı")
            return 3 if 3 in v.values() else (1 if 1 in v.values() else 0)
        return None

    @staticmethod
    def _durdu_satiri(x: str, j: dict) -> str:
        rap = ((j.get("kapi_raporlari") or [{}])[-1]).get("rapor", "?")
        return (f"KUYRUK-DURDU {x} asama={j['durduruldu_asama']} basarili={j.get('basarili')}/{j.get('toplam')} "
                f"rapor={Path(str(rap)).name} → teşhis (beklenti-ihlali.md)")

    def aile_ozet(self, aile: list):
        satir_a3 = set()
        for x in aile[:1]:
            p = self.kok / "kuyruk" / f"{x}.jsonl"
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        j = json.loads(line)
                        if int(j.get("asama", 3)) == 3:
                            satir_a3.add((j.get("varyant"), str(j.get("seed"))))
        ok, bas = set(), 0
        runs = self.kok / "runs"
        for d in runs.iterdir() if runs.exists() else []:
            c = oku_json(d / "config.json") if d.is_dir() else None
            if not c or c.get("kuyruk") not in aile:
                continue
            dn = oku_json(d / ".done") if (d / ".done").exists() else None
            if dn is None:
                continue
            if dn.get("rc") == 0 and int(c.get("asama", 3)) == 3:
                ok.add((c.get("varyant"), str(c.get("seed"))))
            elif dn.get("rc") != 0:
                bas += 1
        return len(ok & satir_a3), len(satir_a3), bas

    def asili(self, birim: str, kuyruklar: list, G: dict | None = None) -> list:
        out = []
        G = self.gruplar() if G is None else G
        simdi = self.saat()
        aday = []
        for p, cj in self.canli_dosyalari():
            kapsam = self.canli_kapsam(cj, G)
            if kapsam is None or kapsam & set(kuyruklar):
                aday.append((float(cj.get("kalp_epoch") or 0), p, cj))
        if not aday:
            if self.coklu and simdi - self.baslangic > self.asili_sn and simdi - self.son_canli_yok.get(birim, 0) > self.asili_sn:
                self.son_canli_yok[birim] = simdi
                out.append(f"CANLI-YOK {birim}: izleme {(simdi - self.baslangic) / 60:.0f} dk'dır sürüyor, bu kuyruğun sekmesinden "
                           f"kalp atışı yok (durum/canli_*.json) → sekme açıldı mı / hücre 7'ye geldi mi?")
            return out
        kalp, p, cj = max(aday, key=lambda x: x[0])
        if cj.get("bitti"):
            return out
        yas = simdi - kalp
        sekme = f" sekme {cj.get('sekme')}" if cj.get("sekme") else ""
        if yas > self.asili_sn:
            if simdi - self.son_asili.get(birim, 0) > self.asili_sn:
                self.son_asili[birim] = simdi
                ne = f" ({birim})" if self.coklu else ""
                out.append(f"ASILI {p.name}{ne} kalp atışı {yas / 60:.0f} dk bayat (oturum {cj.get('oturum')}{sekme}, son {cj.get('zaman')}) "
                           f"→ Colab oturumu/sekme kopmuş olabilir; kullanıcıya:{sekme or ''} defteri yeniden 'Tümünü çalıştır' (devam)")
            return out
        for a in cj.get("aktif") or []:
            if a.get("kuyruk") and not aile_uyesi(a.get("kuyruk"), kuyruklar):
                continue
            hb = a.get("hb_yas_sn")
            if isinstance(hb, (int, float)) and hb > 600:
                rid = a.get("run_id")
                if simdi - self.son_kosum_asili.get(rid, 0) > self.asili_sn:
                    self.son_kosum_asili[rid] = simdi
                    out.append(f"ASILI {rid} Colab-tarafı kalp atışı {hb:.0f}s bayat (epoch {a.get('epoch')}/{a.get('epoch_toplam')}"
                               f"{', ' + sekme.strip() if sekme else ''})")
        return out


def calistir(a) -> int:
    kok = Path(a.drive)
    if not kok.exists():
        print(f"HATA: Drive klasörü yok: {kok}")
        return 2
    kuyruklar = [x.strip() for x in a.kuyruklar.split(",") if x.strip()] if a.kuyruklar else None
    iz = Izleyici(kok, a.grup, a.asili_dk, kuyruklar=kuyruklar)
    if kuyruklar:
        yok = [q for q in kuyruklar if not (kok / "kuyruk" / f"{q}.jsonl").exists()]
        if yok:
            print(f"UYARI: Drive'da kuyruk dosyası yok: {yok} (ad yanlış mı / eşitleme bitmedi mi?)")
    olaylar, rc = iz.tara(ilk=True)
    n_run = len(iz.gorulen_run)
    print(f"IZLE {iz.grup} başladı {datetime.now():%Y-%m-%d %H:%M:%S} · kip {'paralel kuyruklar' if iz.coklu else 'grup'} · "
          f"kuyruklar {iz.kuyruklar()} · mevcut koşum {n_run} "
          f"(bitmiş {sum(1 for v in iz.gorulen_run.values() if v == 'BITTI')}) · hata günlüğü {len(iz.gorulen_hata)} · "
          f"aralık {a.aralik}s · asılı eşiği {a.asili_dk} dk")
    if iz.coklu:
        G = iz.gruplar()
        for q in kuyruklar:
            aday = [(float(cj.get("kalp_epoch") or 0), p.name, cj) for p, cj in iz.canli_dosyalari()
                    if (iz.canli_kapsam(cj, G) or set()) & {q}]
            if aday:
                k, ad, cj = max(aday, key=lambda x: x[0])
                print(f"  {q}: {ad} sekme {cj.get('sekme')} kalp {time.time() - k:.0f}s önce{' (bitti)' if cj.get('bitti') else ''}")
            else:
                print(f"  {q}: henüz canlı dosya yok")
    for o in olaylar:
        print(o)
    if rc is not None:
        return rc
    while True:
        time.sleep(a.aralik)
        olaylar, rc = iz.tara()
        for o in olaylar:
            print(o)
        if rc is not None:
            return rc


# ───────────────────────── kendi-testi (sahte damgalar) ─────────────────────────
def kendi_testi() -> int:
    t = Path(tempfile.mkdtemp(prefix="colab_izle_test_"))
    try:
        (t / "kuyruk").mkdir(); (t / "runs" / "kuyruklar").mkdir(parents=True); (t / "durum").mkdir()
        (t / "kuyruk" / "gruplar.json").write_text(json.dumps({"gruplar": {"g": ["q"]}}), encoding="utf-8")
        satir = lambda v, s, k: json.dumps({"varyant": v, "seed": str(s), "asama": k, "komut": ["x"]})  # noqa: E731
        (t / "kuyruk" / "q.jsonl").write_text("\n".join([satir("a_duman", 0, 1), satir("a", 0, 3), satir("a", 1, 3)]) + "\n",
                                              encoding="utf-8")
        saat = [1000.0]
        iz = Izleyici(t, "g", asili_dk=15, saat=lambda: saat[0])
        def kosum(rid, v, s, k, kuyruk="q", kok=t):
            d = kok / "runs" / rid; d.mkdir()
            (d / "config.json").write_text(json.dumps({"run_id": rid, "varyant": v, "seed": str(s), "asama": k, "kuyruk": kuyruk}),
                                          encoding="utf-8")
            return d
        def bitir(d, rc, va=0.8):
            (d / ".done").write_text(json.dumps({"rc": rc, "sure_sn": 12.0, "metrics": {"val_acc": va} if rc == 0 else None}),
                                     encoding="utf-8")
        def canli(epoch_, aktif=()):
            (t / "durum" / "canli.json").write_text(json.dumps({"kalp_epoch": epoch_, "zaman": "x", "oturum": "o",
                                                               "aktif": list(aktif)}), encoding="utf-8")
        beklenen = []
        o, rc = iz.tara(ilk=True); assert o == [] and rc is None, (o, rc)
        d1 = kosum("r1_a_duman_s0", "a_duman", 0, 1); canli(1000.0)
        o, rc = iz.tara(); assert any(x.startswith("BAŞLADI r1_a_duman_s0") for x in o) and rc is None, o; beklenen += o
        bitir(d1, 0, 0.71)
        with open(t / "runs" / "BILDIRIM.log", "a", encoding="utf-8") as f:
            f.write("2026-10-11T12:00:00 AŞAMA-GEÇTİ q asama=1 rc=0 rapor=x\n2026-10-11T12:00:00 AŞAMA-GEÇTİ baska asama=1\n")
        o, rc = iz.tara(); assert any(x.startswith("BİTTİ r1_a_duman_s0 rc=0") and "val_acc=0.7100" in x for x in o), o
        assert any(x.startswith("AŞAMA-GEÇTİ q") for x in o) and not any("baska" in x for x in o), o; beklenen += o
        d2 = kosum("r2_a_s0", "a", 0, 3)
        saat[0] = 1000.0 + 16 * 60          # canli 16 dk bayat → ASILI
        o, rc = iz.tara(); assert any(x.startswith("ASILI canli.json") for x in o), o; beklenen += o
        o, rc = iz.tara(); assert not any(x.startswith("ASILI") for x in o), o        # tekrar etmez (eşik içinde)
        canli(saat[0], [{"run_id": "r2_a_s0", "hb_yas_sn": 700, "epoch": 3, "epoch_toplam": 30}])
        o, rc = iz.tara(); assert any(x.startswith("ASILI r2_a_s0 Colab-tarafı") for x in o), o; beklenen += o
        (t / "durum" / "hata_20261011-120000.log").write_text("oturum o · kuyruk g\n--- traceback ---\nAssertionError: L4 değil\n",
                                                              encoding="utf-8")
        o, rc = iz.tara(); assert any(x.startswith("HATA-LOG") and "L4 değil" in x for x in o), o; beklenen += o
        # oturum koptu → devam kuyruğu; yetim + devam satırı
        bitir(d2, 96)
        (t / "kuyruk" / "q-devam1.jsonl").write_text(satir("a", 0, 3) + "\n" + satir("a", 1, 3) + "\n", encoding="utf-8")
        with open(t / "runs" / "BILDIRIM.log", "a", encoding="utf-8") as f:
            f.write("2026-10-11T12:30:00 KUYRUK-KOPTU q → devam q-devam1 kalan=2\n")
        o, rc = iz.tara(); assert any("rc=96" in x for x in o) and any(x.startswith("KUYRUK-KOPTU q") for x in o), o; beklenen += o
        d3 = kosum("r3_a_s0", "a", 0, 3, "q-devam1"); d4 = kosum("r4_a_s1", "a", 1, 3, "q-devam1")
        bitir(d3, 0, 0.86); bitir(d4, 0, 0.87)
        o, rc = iz.tara(); assert rc is None, (o, rc); beklenen += o
        (t / "runs" / "kuyruklar" / "q-devam1.KUYRUK.done").write_text(json.dumps({"toplam": 2, "basarili": 2, "basarisiz": 0,
                                                                                   "durduruldu_asama": None}), encoding="utf-8")
        o, rc = iz.tara(); assert any(x.startswith("KUYRUK-BİTTİ q") and "2/2" in x for x in o) and rc == 1, (o, rc); beklenen += o
        # DURDU senaryosu
        iz2 = Izleyici(t, "g2")
        (t / "kuyruk" / "gruplar.json").write_text(json.dumps({"gruplar": {"g": ["q"], "g2": ["z"]}}), encoding="utf-8")
        (t / "kuyruk" / "z.jsonl").write_text(satir("b", 0, 1) + "\n", encoding="utf-8")
        (t / "runs" / "kuyruklar" / "z.KUYRUK.done").write_text(json.dumps(
            {"toplam": 1, "basarili": 1, "durduruldu_asama": 1, "kapi_raporlari": [{"rapor": "sonuclar/beklenti-z-asama1.md"}]}),
            encoding="utf-8")
        o, rc = iz2.tara(ilk=True); assert rc == 3 and any(x.startswith("KUYRUK-DURDU z asama=1") for x in o), (o, rc)
        beklenen += o
        print("KENDİ-TESTİ (v1 grup kipi) GEÇTİ — üretilen olaylar:")
        for x in beklenen:
            print("  " + x)
    finally:
        shutil.rmtree(t, ignore_errors=True)
    return kendi_testi_paralel()


def kendi_testi_paralel() -> int:
    """v2: üç paralel sekme (q1 A, q2 B, q3 C) + eski tek-oturum dosyaları (canli.json, BILDIRIM.log) aynı Drive'da."""
    t = Path(tempfile.mkdtemp(prefix="colab_izle_test2_"))
    try:
        (t / "kuyruk").mkdir(); (t / "runs" / "kuyruklar").mkdir(parents=True); (t / "durum").mkdir()
        (t / "kuyruk" / "gruplar.json").write_text(json.dumps({"gruplar": {"kc1": ["kc1"], "g": ["q1", "q2", "q3"]}}),
                                                   encoding="utf-8")
        satir = lambda v, s, k: json.dumps({"varyant": v, "seed": str(s), "asama": k, "komut": ["x"]})  # noqa: E731
        for q in ("q1", "q2", "q3", "kc1"):
            (t / "kuyruk" / f"{q}.jsonl").write_text("\n".join([satir(f"{q}_duman", 0, 1), satir(q, 0, 3)]) + "\n", encoding="utf-8")
        saat = [5000.0]
        iz = Izleyici(t, asili_dk=15, saat=lambda: saat[0], kuyruklar=["q1", "q2", "q3"])

        def kosum(rid, v, s, k, kuyruk):
            d = t / "runs" / rid; d.mkdir()
            (d / "config.json").write_text(json.dumps({"run_id": rid, "varyant": v, "seed": str(s), "asama": k, "kuyruk": kuyruk}),
                                          encoding="utf-8")
            return d

        def bitir(d, rc, va=0.8):
            (d / ".done").write_text(json.dumps({"rc": rc, "sure_sn": 12.0, "metrics": {"val_acc": va} if rc == 0 else None}),
                                     encoding="utf-8")

        def canli(oturum, sekme, kuyruk, epoch_, aktif=(), bitti=False):
            j = {"kalp_epoch": epoch_, "zaman": "x", "oturum": oturum, "grup": kuyruk, "kuyruklar": [kuyruk], "sekme": sekme,
                 "aktif": list(aktif)}
            if bitti:
                j["bitti"] = True
            (t / "durum" / f"canli_{oturum}.json").write_text(json.dumps(j), encoding="utf-8")

        def damga(ad, durdu=None):
            (t / "runs" / "kuyruklar" / f"{ad}.KUYRUK.done").write_text(json.dumps(
                {"toplam": 2, "basarili": 2, "durduruldu_asama": durdu,
                 "kapi_raporlari": [{"rapor": f"sonuclar/beklenti-{ad}-asama{durdu}.md"}] if durdu else []}), encoding="utf-8")

        beklenen = []
        # eski tek oturum (kc1) dosyaları: bayat canli.json + BILDIRIM.log → izlenen kuyruklara KARIŞMAMALI
        (t / "durum" / "canli.json").write_text(json.dumps({"kalp_epoch": 1.0, "oturum": "eski", "grup": "kc1",
                                                           "aktif": [{"run_id": "x", "kuyruk": "kc1", "hb_yas_sn": 9999}]}),
                                                encoding="utf-8")
        o, rc = iz.tara(ilk=True); assert o == [] and rc is None, (o, rc)
        canli("A-1", "A", "q1", saat[0]); canli("B-1", "B", "q2", saat[0])
        r1 = kosum("r1_q1_duman_s0", "q1_duman", 0, 1, "q1"); r2 = kosum("r2_q2_duman_s0", "q2_duman", 0, 1, "q2")
        with open(t / "runs" / "BILDIRIM_A-1.log", "a", encoding="utf-8") as f:
            f.write("2026-10-11T14:00:00 KUYRUK-BAŞLADI q1 toplam=2\n")
        with open(t / "runs" / "BILDIRIM_B-1.log", "a", encoding="utf-8") as f:
            f.write("2026-10-11T14:00:01 KUYRUK-BAŞLADI q2 toplam=2\n")
        with open(t / "runs" / "BILDIRIM.log", "a", encoding="utf-8") as f:
            f.write("2026-10-11T14:00:02 AŞAMA-GEÇTİ kc1 asama=2 rc=0\n")
        o, rc = iz.tara()
        assert sum(x.startswith("BAŞLADI r") for x in o) == 2 and sum(x.startswith("KUYRUK-BAŞLADI q") for x in o) == 2, o
        assert not any("kc1" in x for x in o) and not any(x.startswith("ASILI") for x in o) and rc is None, o; beklenen += o
        # 16 dk sonra: A bayat → ASILI (q1, sekme A); B taze; q3'ün hiç canlı dosyası yok → CANLI-YOK q3
        saat[0] += 16 * 60
        canli("B-1", "B", "q2", saat[0], [{"run_id": "r2_q2_duman_s0", "kuyruk": "q2", "hb_yas_sn": 20},
                                          {"run_id": "baska", "kuyruk": "q1", "hb_yas_sn": 999}])
        o, rc = iz.tara()
        assert any(x.startswith("ASILI canli_A-1.json (q1)") and "sekme A" in x for x in o), o
        assert any(x.startswith("CANLI-YOK q3") for x in o) and not any("baska" in x for x in o), o
        assert not any("canli_B-1" in x for x in o) and not any("canli.json" in x for x in o), o; beklenen += o
        # A sekmesi yeniden açıldı: yeni oturum dosyası eskisini geçer → ASILI yok; yetim + koptu bildirimi
        canli("A-2", "A", "q1", saat[0]); bitir(r1, 96)
        with open(t / "runs" / "BILDIRIM_A-2.log", "a", encoding="utf-8") as f:
            f.write("2026-10-11T14:20:00 YETİM-DAMGA r1_q1_duman_s0 rc=96 (Colab oturumu koptu)\n"
                    "2026-10-11T14:20:00 KUYRUK-KOPTU q1 → devam q1-devam1 kalan=2\n")
        (t / "kuyruk" / "q1-devam1.jsonl").write_text(satir("q1_duman", 0, 1) + "\n" + satir("q1", 0, 3) + "\n", encoding="utf-8")
        o, rc = iz.tara()
        assert any(x.startswith("YETİM-DAMGA r1") for x in o) and any(x.startswith("KUYRUK-KOPTU q1") for x in o), o
        assert not any(x.startswith("ASILI canli_A") for x in o), o; beklenen += o
        # q2 aşama kapısında DURDU → izleyici ÇIKMAZ
        bitir(r2, 0, 0.5); damga("q2", durdu=1)
        o, rc = iz.tara(); assert any(x.startswith("KUYRUK-DURDU q2 asama=1") for x in o) and rc is None, (o, rc); beklenen += o
        saat[0] += 40 * 60     # B'nin canlısı bayatladı ama q2 bitmiş sayılır → ASILI yok; A da bayat → ASILI q1
        o, rc = iz.tara(); assert not any("q2" in x for x in o) and any("(q1)" in x for x in o), o
        # q1 (devam1) bitti; q3 sekmesi geç açıldı ve bitti
        r3 = kosum("r3_q1_s0", "q1", 0, 3, "q1-devam1"); bitir(r3, 0, 0.9); damga("q1-devam1")
        canli("C-1", "C", "q3", saat[0]); r4 = kosum("r4_q3_s0", "q3", 0, 3, "q3"); bitir(r4, 0, 0.88)
        o, rc = iz.tara(); assert any(x.startswith("KUYRUK-BİTTİ q1") and "son=q1-devam1" in x for x in o) and rc is None, (o, rc)
        beklenen += o
        damga("q3"); canli("C-1", "C", "q3", saat[0], bitti=True)
        o, rc = iz.tara()
        assert any(x.startswith("KUYRUK-BİTTİ q3") for x in o) and any(x.startswith("TÜMÜ-BİTTİ") for x in o) and rc == 3, (o, rc)
        beklenen += o
        print("KENDİ-TESTİ (v2 paralel kuyruk kipi) GEÇTİ — üretilen olaylar:")
        for x in beklenen:
            print("  " + x)
        return 0
    finally:
        shutil.rmtree(t, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grup", default="kc1-tekrar")
    ap.add_argument("--kuyruklar", default=None, help="virgülle ayrık kuyruk adları (paralel sekmeler; hepsi bitince çık)")
    ap.add_argument("--drive", default=VARSAYILAN_DRIVE)
    ap.add_argument("--aralik", type=float, default=60)
    ap.add_argument("--asili-dk", type=float, default=20)
    ap.add_argument("--kendi-testi", action="store_true")
    a = ap.parse_args()
    if a.kendi_testi:
        return kendi_testi()
    if a.asili_dk < 15:
        print("UYARI: Drive eşitleme gecikmesi nedeniyle asılı eşiği 15 dk'ya yükseltildi")
        a.asili_dk = 15
    return calistir(a)


if __name__ == "__main__":
    sys.exit(main())
