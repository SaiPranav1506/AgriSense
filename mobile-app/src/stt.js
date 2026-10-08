// Speech-to-text (STT) shim.
//
// IMPORTANT (Expo Go constraint):
//   expo-speech is TEXT-TO-SPEECH ONLY — it has no speech-recognition API.
//   The Expo STT package, expo-speech-recognition, is a NATIVE module that
//   requires a development build (expo prebuild / dev-client). It does NOT
//   run inside stock Expo Go.
//
// So for the Expo-Go demo we use the device's built-in VOICE KEYBOARD
// (dictation) — the user taps the mic, the input focuses, and they tap the
// keyboard's mic to dictate. No native module needed, works in Expo Go.
//
// To upgrade to true in-app STT later (development build):
//   npx expo install expo-speech-recognition
//   npx expo prebuild            # generate native projects
//   npx expo run:android         # build & install a dev client on the phone
// then swap startListening() below for the expo-speech-recognition hook.

export const STT_MODE = 'keyboard-dictation'; // 'expo-speech-recognition' after a dev build

// Returns true if real in-app STT is available in this runtime.
export function isNativeSttAvailable() {
  return false; // stock Expo Go
}

// In Expo Go this is a no-op placeholder; the chat screen focuses the input
// and the farmer uses the keyboard's microphone. Kept as an async function
// so a dev-build implementation can be dropped in without changing callers.
export async function startListening({ onResult } = {}) {
  // no-op on Expo Go — see comment above.
  return () => {};
}
