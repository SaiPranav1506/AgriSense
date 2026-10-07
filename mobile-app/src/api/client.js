// Thin axios wrapper around the AgriSense backend, matching the
// exact request/response contract discovered from the Flask app.
import axios from 'axios';
import { ENDPOINTS } from '../config/api';

const http = axios.create({ timeout: 30000 });

// ── Crop recommendation ──────────────────────────────────────────────
// body: { N, P, K, temperature, humidity, ph, rainfall, lat?, lon? }
// -> { recommended_crop, confidence, top3: [{crop, confidence}] }
export async function predictCrop(params) {
  const { data } = await http.post(ENDPOINTS.crop, params);
  return data;
}

// ── Yield prediction ─────────────────────────────────────────────────
// body: { area, crop, year, rainfall, pesticides, avg_temp, area_ha, lat?, lon? }
// -> { yield_per_ha, total_yield, unit }
export async function predictYield(params) {
  const { data } = await http.post(ENDPOINTS.yield, params);
  return data;
}

// ── Disease detection (multipart image upload) ───────────────────────
// asset: { uri } -> { disease_class, confidence, top3_classes: [{class, confidence}] }
export async function predictDisease(imageAsset) {
  const form = new FormData();
  if (imageAsset.file) {
    // Web: append the real File object so the browser sets content + filename.
    form.append('image', imageAsset.file, imageAsset.fileName || 'leaf.jpg');
  } else {
    // React Native: pass {uri, name, type} directly in FormData.
    form.append('image', {
      uri: imageAsset.uri,
      name: imageAsset.fileName || 'leaf.jpg',
      type: imageAsset.mimeType || 'image/jpeg',
    });
  }
  const { data } = await http.post(ENDPOINTS.disease, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 60000,
  });
  return data;
}

// ── Chat agent ───────────────────────────────────────────────────────
// messages: [{role:'user'|'assistant', content:string}] -> { response }
export async function sendChat(messages) {
  const { data } = await http.post(ENDPOINTS.chat, { messages }, { timeout: 60000 });
  return data;
}

// ── Helpers ──────────────────────────────────────────────────────────
export async function getHealth() {
  const { data } = await http.get(ENDPOINTS.health, { timeout: 10000 });
  return data;
}

export async function getEncoders() {
  const { data } = await http.get(ENDPOINTS.encoders);
  return data;
}
