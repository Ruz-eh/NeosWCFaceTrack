REM pyinstaller --onefile ^
REM     --add-binary dshowcapture/dshowcapture_x86.dll;. ^
REM     --add-binary dshowcapture/dshowcapture_x64.dll;. ^
REM     facetracker.py

pyinstaller --onedir ^
    --add-binary dshowcapture/dshowcapture_x86.dll;. ^
    --add-binary dshowcapture/dshowcapture_x64.dll;. ^
    facetracker.py