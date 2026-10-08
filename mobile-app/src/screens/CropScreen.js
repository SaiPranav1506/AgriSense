import React, { useState } from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { Screen, Title, Card, Field, Button, ErrorBox, ResultRow, COLORS } from '../components/ui';
import { SafeScreen } from '../components/AndroidChrome';
import { predictCrop } from '../api/client';

const DEFAULTS = {
  N: '90', P: '42', K: '43',
  temperature: '21', humidity: '82', ph: '6.5', rainfall: '202',
  lat: '', lon: '',
};

export default function CropScreen() {
  const [form, setForm] = useState(DEFAULTS);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  const set = (k) => (v) => setForm((f) => ({ ...f, [k]: v }));

  async function submit() {
    setError('');
    setResult(null);
    const num = (s) => parseFloat(s);
    const body = {
      N: num(form.N), P: num(form.P), K: num(form.K),
      temperature: num(form.temperature), humidity: num(form.humidity),
      ph: num(form.ph), rainfall: num(form.rainfall),
    };
    if (form.lat && form.lon) {
      body.lat = num(form.lat);
      body.lon = num(form.lon);
    }
    for (const k of ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']) {
      if (Number.isNaN(body[k])) {
        setError(`Please enter a valid number for ${k}.`);
        return;
      }
    }
    setLoading(true);
    try {
      setResult(await predictCrop(body));
    } catch (e) {
      setError(e.response?.data?.error || e.message || 'Request failed');
    } finally {
      setLoading(false);
    }
  }

  return (
    <SafeScreen>
      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ padding: 16, paddingBottom: 96 }}>
        <Card>
          <View style={styles.row}>
            <View style={styles.col}><Field label="Nitrogen (N)" value={form.N} onChangeText={set('N')} /></View>
            <View style={styles.col}><Field label="Phosphorus (P)" value={form.P} onChangeText={set('P')} /></View>
            <View style={styles.col}><Field label="Potassium (K)" value={form.K} onChangeText={set('K')} /></View>
          </View>
          <View style={styles.row}>
            <View style={styles.col}><Field label="Temp (°C)" value={form.temperature} onChangeText={set('temperature')} /></View>
            <View style={styles.col}><Field label="Humidity (%)" value={form.humidity} onChangeText={set('humidity')} /></View>
          </View>
          <View style={styles.row}>
            <View style={styles.col}><Field label="Soil pH" value={form.ph} onChangeText={set('ph')} /></View>
            <View style={styles.col}><Field label="Rainfall (mm)" value={form.rainfall} onChangeText={set('rainfall')} /></View>
          </View>
          <View style={styles.row}>
            <View style={styles.col}><Field label="Latitude (optional)" value={form.lat} onChangeText={set('lat')} /></View>
            <View style={styles.col}><Field label="Longitude (optional)" value={form.lon} onChangeText={set('lon')} /></View>
          </View>
          <Button title="Recommend Crop" onPress={submit} loading={loading} />
        </Card>

        <ErrorBox message={error} />

        {result && (
          <Card>
            <Text style={styles.section}>Top recommendation</Text>
            <ResultRow label="Crop" value={result.recommended_crop} strong />
            <ResultRow label="Confidence" value={`${result.confidence}%`} />
            {Array.isArray(result.top3) && (
              <>
                <Text style={[styles.section, { marginTop: 14 }]}>Top 3 matches</Text>
                {result.top3.map((t, i) => (
                  <ResultRow key={i} label={`#${i + 1} ${t.crop}`} value={`${t.confidence}%`} />
                ))}
              </>
            )}
          </Card>
        )}
      </ScrollView>
    </SafeScreen>
  );
}

const styles = StyleSheet.create({
  row: { flexDirection: 'row', marginHorizontal: -4 },
  col: { flex: 1, paddingHorizontal: 4 },
  section: { fontSize: 11, fontWeight: '700', color: COLORS.muted, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1.5 },
});
