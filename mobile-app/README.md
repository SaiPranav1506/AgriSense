# AgriSense Mobile App

Pure-client React Native (Expo) app for the AgriSense backend. It does **not**
depend on the server codebase — it only calls its HTTP API.

## Screens
- **Home** — dashboard + backend health status
- **Crop Recommendation** — N, P, K, pH, temp, humidity, rainfall → top-3 crops
- **Yield Prediction** — region/crop/climate → tonnes per hectare
- **Disease Detection** — camera/gallery leaf photo → diagnosis + confidence
- **Ask AgriSense** — chat with the farm assistant (mic via voice keyboard, TTS read-aloud)

## Setup

```bash
cd mobile-app
npm install
```

## Configure the backend URL

Copy `.env.example` to `.env` and set `EXPO_PUBLIC_API_URL`:

```bash
cp .env.example .env
```

For a **physical Android phone (iQOO 15) on the same Wi-Fi** as your PC, use your
PC's local IP (find it with `ipconfig` → IPv4 Address):

```
EXPO_PUBLIC_API_URL=http://192.168.1.5:5000
```

| Run target       | Value                              |
|------------------|------------------------------------|
| Physical phone   | `http://<PC-LAN-IP>:5000`          |
| Android emulator | `http://10.0.2.2:5000`             |
| Web / same PC    | `http://localhost:5000`            |

## Run

```bash
npm start
```

Then open with **Expo Go** on the phone and scan the QR code (same Wi-Fi).

## Backend requirements

The Flask server must be reachable from the phone:

```bash
# from the repo root
python app/app.py      # serves on 0.0.0.0:5000
```

Make sure `ANTHROPIC_API_KEY` is set for the chat endpoint (the app also works
with the server's demo-mode fallback if it isn't).

## Note on the mic (speech-to-text)

`expo-speech` is text-to-speech only. True on-device STT (`expo-speech-recognition`)
needs a **development build**, not stock Expo Go. The chat mic therefore uses the
phone's **voice keyboard dictation**, which works in Expo Go today. See
`src/stt.js` for the one-line upgrade path.
