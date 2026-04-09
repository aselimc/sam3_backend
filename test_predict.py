import json
import os
import urllib.request

base = "//gtcsynology2/GISTeamShared/Baris/03_Atlas_University/Kat2"
images = ["p000065.jpg", "p000070.jpg", "p000075.jpg"]
output_dir = r"C:\Users\aselimc\workdir\vision\sam3_backend\data"

for img in images:
    path = os.path.join(base, img)
    payload = json.dumps({
        "image_path": path,
        "queries": ["human"],
        "output_dir": output_dir,
    }).encode()
    req = urllib.request.Request(
        "http://localhost:8000/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        print(f"{img}: {resp.status}")
        print(resp.read().decode()[:500])
    except Exception as e:
        print(f"{img}: ERROR - {e}")
        if hasattr(e, "read"):
            print(e.read().decode()[:500])
    print()
