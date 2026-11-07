# Verilen kaynak dosyayı, __file__ = WS/varyantlar/dinov2_ft/ince_ayar_vit.py olarak çalıştırır
# (eski ve yeni sürüm aynı içe aktarma yolunu, aynı ortak/ ve dinov2_ozn/ modüllerini kullanır).
import hashlib, sys
from pathlib import Path
kaynak, sahte = Path(sys.argv[1]), Path(sys.argv[2])
src = kaynak.read_bytes()
print(f"[kaynak_kos] {kaynak} sha256={hashlib.sha256(src).hexdigest()}", flush=True)
sys.argv = [str(sahte)] + sys.argv[3:]
sys.path.insert(0, str(sahte.parent))
g = {"__name__": "__main__", "__file__": str(sahte), "__builtins__": __builtins__}
exec(compile(src, str(sahte), "exec"), g)
