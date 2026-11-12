# Virtual Pointer

A projekt két kamera képéből becsüli a mutatóujj térbeli irányát, kiszámítja a táblával való metszést, majd a találatot egy átlátszó, kattintást átengedő Windows overlayen mutatja. A `szam_lat-34-39.pdf` a matematikai és funkcionális tervet tartalmazza.

## Telepítés és demó

Python 3.12 ajánlott. Windows PowerShellben:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python virtual_pointer.py --mock
```

A `--mock` szintetikus kameraképekkel és kalibrációval végigfuttatja a teljes geometriai láncot; kamerák nélkül is látható a mozgó piros pointer. A `calibration_params.example.json` csak formátumminta, valódi kamerához nem használható.

## Valódi kamerák előkészítése

1. Rögzíts két kamerát úgy, hogy ugyanazt a kezet és táblát lássák. A kalibráció után ne mozdítsd el őket.
2. Nyomtass egy sík sakktáblamintát, és mérd meg egy négyzet oldalát milliméterben. Az alapértelmezett minta **9 × 6 belső sarok**, a mintán tehát 10 × 7 négyzet van. Ha más mintát használsz, add meg a `--cols` és `--rows` értékét mindkét parancsnál.
3. Gyűjts legalább 20–25 éles képpárt változó távolságból és dőlésszögből, a kép különböző területein. A SPACE csak akkor ment, ha a minta mindkét kamerán látszik; Q kilép.

```powershell
python stereo_calibrate.py capture --camera1 0 --camera2 1 --count 25 --cols 9 --rows 6
python stereo_calibrate.py calibrate --cols 9 --rows 6 --square-mm 25
```

A második parancs előállítja a `calibration_params.json` fájlt (K1, D1, K2, D2, R, T, képméret és RMS hibák). Nézd meg a kiírt sztereó RMS hibát; 1 px felett érdemes változatosabb, élesebb képpárokat készíteni. A `calibration_images/` és a mért paraméterek helyi fájlok, nem kerülnek automatikusan Gitbe.

4. Állítsd be a valódi tábla szélességét és magasságát. A mellékelt `board_markers.png` a négy ArUco jelölőt tartalmazza az alap 600 × 400 mm táblához; más mérethez generálj újat az alábbi paranccsal. A fekete jelölők 80 mm szélesek, és a tábla szélétől 20 mm-re kezdődnek. **Nyomtasd tényleges méretben, átméretezés nélkül**; ellenőrizd vonalzóval. A kamera 1 lássa mind a négy jelölőt, a két kamera pedig ugyanazt a mutatóujjat.

```powershell
python virtual_pointer.py --generate-board board_markers.png --board-width-mm 600 --board-height-mm 400 --marker-size-mm 80
python virtual_pointer.py --camera1 0 --camera2 1 --screen 0 --board-width-mm 600 --board-height-mm 400 --marker-size-mm 80
```

Eltérő helyen tárolt kalibrációhoz használd a `--calibration` opciót. Ha a vetítő második kijelző, állítsd a `--screen` értékét a megfelelő Qt kijelzőindexre; az overlay az egész virtuális asztalt lefedi. A program a kamerák aktuális felbontására skálázza a belső paramétereket, de az eltérő képkivágás vagy képarány ronthatja a pontosságot. Valódi, számszerű pontosságot csak a konkrét kamerákon, táblán és vetítőn végzett méréssel lehet igazolni.

A két USB-kamera képei szoftveresen közel egy időben készülnek (`grab` mindkettőn, majd `retrieve`); gyors kézmozgásnál a hardveres szinkron hiánya hibát okozhat.

## A feldolgozás menete

`stereo_calibrate.py` páros sakktáblaképekből OpenCV-vel kalibrálja a kamerákat. Futás közben a MediaPipe mindkét képen megkeresi a mutatóujj pontjait, az OpenCV torzításkorrekció és trianguláció 3D pontokat ad, a DIP→TIP vektor sugara metszi az ArUco jelölőkből becsült táblasíkot. A tábla helyi koordinátái képernyőpixelekké alakulnak, a Kalman-szűrő csillapítja a remegést, a Qt overlay pedig kirajzolja a találatot.
