// Central place that resolves the backend base URL.
// Priority: EXPO_PUBLIC_API_URL env var -> app.json extra -> localhost fallback.
import Constants from 'expo-constants';

const fromEnv = process.env.EXPO_PUBLIC_API_URL;
const fromExtra = Constants.expoConfig?.extra?.apiUrl;

export const API_URL = (fromEnv || fromExtra || 'http://localhost:5000').replace(/\/$/, '');

// Desktop web UI (the Vite/React frontend). Used by the shell's Desktop mode on web.
const frontendEnv = process.env.EXPO_PUBLIC_FRONTEND_URL;
export const FRONTEND_URL = (
  frontendEnv || Constants.expoConfig?.extra?.frontendUrl || 'http://localhost:5173'
).replace(/\/$/, '');

export const ENDPOINTS = {
  health: `${API_URL}/health`,
  crop: `${API_URL}/predict-crop`,
  cropFamily: `${API_URL}/predict-crop-family`,
  yield: `${API_URL}/predict-yield`,
  disease: `${API_URL}/predict-disease`,
  chat: `${API_URL}/api/chat`,
  encoders: `${API_URL}/encoders`,
};
