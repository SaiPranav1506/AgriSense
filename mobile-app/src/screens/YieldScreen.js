import React, { useEffect, useState } from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { Screen, Title, Card, Field, Button, ErrorBox, ResultRow, COLORS } from '../components/ui';
import { SafeScreen } from '../components/AndroidChrome';
import { predictYield, getEncoders } from '../api/client';

const DEFAULTS = {
  area: 'India', crop: 'Rice, paddy', year: '2024',
  rainfall: '1000', pesticides: '100', avg_temp: '24', area_ha: '1.0',
  lat: '', lon: '',
};

export default function YieldScreen() {
  const [form, setForm] = useState(DEFAULTS);
  const [areas, setAreas] = useState([]);
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  useEffect(() => {
    getEncoders()
      .then((d) => { setAreas(d.areas || []); setItems(d.items || []); })
      .catch(() => {});
  }, []);

  const set = (k) => (v) => setForm((f) => ({ ...f, [k]: v }));

  async function submit() {
    setError('');
    setResult(null);
    const body = {
      area: form.area, crop: form.crop,
      year: parseInt(form.year, 10),
      rainfall: parseFloat(form.rainfall),
      pesticides: parseFloat(form.pesticides),
      avg_temp: parseFloat(form.avg_temp),
      area_ha: parseFloat(form.area_ha),
    };
    if (form.lat && form.lon) {
      body.lat = parseFloat(form.lat);
      body.lon = parseFloat(form.lon);
    }
    setLoading(true);
    try {
      setResult(await predictYield(body));
    } catch (e) {
      setError(e.response?.data?.error || e.message || 'Request failed');
    } finally {
      setLoading(false);
    }
  }

  return (
    <SafeScreen>
      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ padding: 16, paddingBottom: 96 }}>
        <Title>Yield Prediction</Title>
        <Card>
          <Field label="Area / Region" value={form.area} onChangeText={set('area')} keyboardType="default" placeholder={areas[0] || 'India'} />
          <Field label="Crop" value={form.crop} onChangeText={set('crop')} keyboardType="default" placeholder={items[0] || 'Rice, paddy'} />
          <View style={styles.row}>
            <View style={styles.col}><Field label="Year" value={form.year} onChangeText={set('year')} /></View>
            <View style={styles.col}><Field label="Farm area (ha)" value={form.area_ha} onChangeText={set('area_ha')} /></View>
          </View>
          <View style={styles.row}>
            <View style={styles.col}><Field label="Rainfall (mm)" value={form.rainfall} onChangeText={set('rainfall')} /></View>
            <View style={styles.col}><Field label="Avg temp (°C)" value={form.avg_temp} onChangeText={set('avg_temp')} /></View>
          </View>
          <Field label="Pesticides (tonnes)" value={form.pesticides} onChangeText={set('pesticides')} />
          <View style={styles.row}>
            <View style={styles.col}><Field label="Latitude (optional)" value={form.lat} onChangeText={set('lat')} /></View>
            <View style={styles.col}><Field label="Longitude (optional)" value={form.lon} onChangeText={set('lon')} /></View>
          </View>
          <Button title="Predict Yield" onPress={submit} loading={loading} />
        </Card>

        <ErrorBox message={error} />

        {result && (
          <Card>
            <Text style={styles.section}>Forecast</Text>
            <ResultRow label="Yield per hectare" value={`${result.yield_per_ha} ${result.unit}`} strong />
            <ResultRow label="Total (your area)" value={`${result.total_yield} t`} />
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
