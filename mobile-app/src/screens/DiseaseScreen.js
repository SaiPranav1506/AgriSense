import React, { useState } from 'react';
import { Image, ScrollView, StyleSheet, Text, View } from 'react-native';
import { Screen, Title, Card, Button, ErrorBox, ResultRow, COLORS } from '../components/ui';
import { SafeScreen } from '../components/AndroidChrome';
import { predictDisease } from '../api/client';
import { pickImage } from '../components/pickImage';

export default function DiseaseScreen() {
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  async function choose(fromCamera) {
    setError('');
    setResult(null);
    try {
      const asset = await pickImage(fromCamera);
      if (asset) setImage(asset);
    } catch (e) {
      setError(e.message || 'Could not open the image picker.');
    }
  }

  async function analyze() {
    if (!image) return;
    setLoading(true);
    setError('');
    setResult(null);
    try {
      setResult(await predictDisease(image));
    } catch (e) {
      setError(e.response?.data?.error || e.message || 'Request failed');
    } finally {
      setLoading(false);
    }
  }

  const pretty = (s) => (s || '').replace(/___/g, ' — ').replace(/_/g, ' ');

  return (
    <SafeScreen>
      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ padding: 16, paddingBottom: 96 }}>
        <Title>Disease Detection</Title>
        <Text style={styles.sub}>Snap or upload a clear photo of the affected leaf.</Text>
        <Card>
          {image ? (
            <Image source={{ uri: image.uri }} style={styles.preview} />
          ) : (
            <View style={styles.placeholder}>
              <Text style={styles.placeholderEmoji}>🍃</Text>
              <Text style={styles.placeholderText}>No leaf photo selected</Text>
            </View>
          )}
          <View style={styles.btnRow}>
            <View style={{ flex: 1, marginRight: 6 }}>
              <Button title="📷 Camera" variant="secondary" onPress={() => choose(true)} />
            </View>
            <View style={{ flex: 1, marginLeft: 6 }}>
              <Button title="🖼 Upload" variant="secondary" onPress={() => choose(false)} />
            </View>
          </View>
          <View style={{ height: 12 }} />
          <Button title="Analyze Leaf" onPress={analyze} loading={loading} disabled={!image} />
        </Card>

        <ErrorBox message={error} />

        {result && (
          <Card>
            <Text style={styles.section}>Diagnosis</Text>
            <ResultRow label="Condition" value={pretty(result.disease_class)} strong />
            <ResultRow label="Confidence" value={`${result.confidence}%`} />
            {Array.isArray(result.top3_classes) && (
              <>
                <Text style={[styles.section, { marginTop: 14 }]}>Top 3 possibilities</Text>
                {result.top3_classes.map((t, i) => (
                  <ResultRow key={i} label={`#${i + 1} ${pretty(t.class)}`} value={`${t.confidence}%`} />
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
  sub: { color: COLORS.muted, fontSize: 14, marginBottom: 14 },
  preview: { width: '100%', height: 260, borderRadius: 14, marginBottom: 14 },
  placeholder: {
    height: 220, borderRadius: 14, marginBottom: 14,
    backgroundColor: COLORS.cardAlt, alignItems: 'center', justifyContent: 'center',
    borderWidth: 1, borderColor: COLORS.border, borderStyle: 'dashed',
  },
  placeholderEmoji: { fontSize: 40, marginBottom: 8 },
  placeholderText: { color: COLORS.muted },
  btnRow: { flexDirection: 'row' },
  section: {
    fontSize: 11, fontWeight: '700', color: COLORS.muted, marginBottom: 8,
    textTransform: 'uppercase', letterSpacing: 1.5,
  },
});
