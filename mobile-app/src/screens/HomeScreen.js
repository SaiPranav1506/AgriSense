import React, { useEffect, useState } from 'react';
import { Platform, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { Screen, COLORS, RIPPLE } from '../components/ui';
import { getHealth } from '../api/client';
import { API_URL } from '../config/api';
import { useViewMode } from '../components/ResponsiveShell';

const MODULES = [
  { key: 'Crop', emoji: '🌾', label: 'Crop' },
  { key: 'Yield', emoji: '📈', label: 'Yield' },
  { key: 'Disease', emoji: '🍃', label: 'Disease' },
  { key: 'Chat', emoji: '💬', label: 'Assistant' },
];

const STATS = [
  { value: '140M+', label: 'Farmers in India' },
  { value: '58%', label: 'Workforce in agri' },
  { value: '50%', label: 'Annual crop loss' },
  { value: '3', label: 'AI models unified' },
];

export default function HomeScreen({ navigation }) {
  const [health, setHealth] = useState(null);
  const { isMobile, mode, setMode } = useViewMode();

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
    if (Platform.OS === 'web') import('../web/bgEffect').then((m) => m.default?.()).catch(() => {});
  }, []);

  const online = !!health?.models_loaded;

  return (
    <Screen>
      {Platform.OS === 'web' && <View nativeID="bg-canvas-host" style={StyleSheet.absoluteFill} pointerEvents="none" />}
      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scroll}>

        {/* ── TOP: header + summary dashboard ─────────────────────── */}
        <View style={styles.headerRow}>
          <Text style={styles.logo}>AgriSense</Text>
          {/* view-mode toggle (auto / mobile / desktop) */}
          <View style={styles.modeRow}>
            <ModeBtn icon="📱" active={mode === 'mobile'} onPress={() => setMode('mobile')} label="Mobile view" />
            <ModeBtn icon="💻" active={mode === 'desktop'} onPress={() => setMode('desktop')} label="Desktop view" />
          </View>
        </View>
        <Text style={styles.tagline}>SMART CLIMATE AGRICULTURE · AI SYSTEM</Text>

        <View style={styles.badge}>
          <View style={[styles.dot, { backgroundColor: online ? COLORS.good : COLORS.error }]} />
          <Text style={styles.badgeText}>{online ? 'CLIMATE FEED ACTIVE' : 'BACKEND OFFLINE'}</Text>
        </View>

        {/* Stats dashboard */}
        <View style={styles.statsGrid}>
          {STATS.map((s) => (
            <View key={s.label} style={[styles.stat, { width: isMobile ? '48%' : '23%' }]}>
              <Text style={styles.statValue}>{s.value}</Text>
              <Text style={styles.statLabel}>{s.label.toUpperCase()}</Text>
            </View>
          ))}
        </View>

        {/* ── BOTTOM: module grid ─────────────────────────────────── */}
        <Text style={styles.sectionTitle}>MODULES</Text>
        <View style={styles.grid}>
          {MODULES.map((m) => (
            <Pressable
              key={m.key}
              onPress={() => navigation.navigate(m.key)}
              android_ripple={RIPPLE}
              accessibilityRole="button"
              accessibilityLabel={m.label}
              style={({ pressed }) => [
                styles.tile,
                { width: isMobile ? '48%' : '23%' },
                Platform.OS === 'web' && pressed && { opacity: 0.85 },
              ]}
            >
              <View style={styles.tileIconBox}>
                <Text style={styles.tileEmoji}>{m.emoji}</Text>
              </View>
              <Text style={styles.tileLabel}>{m.label}</Text>
            </Pressable>
          ))}
        </View>

        <Text style={styles.footer}>Backend: {API_URL}</Text>
      </ScrollView>
    </Screen>
  );
}

function ModeBtn({ icon, active, onPress, label }) {
  return (
    <Pressable
      onPress={onPress}
      android_ripple={{ ...RIPPLE, borderless: true }}
      accessibilityRole="button"
      accessibilityLabel={label}
      style={[styles.modeBtn, active && styles.modeBtnActive]}
    >
      <Text style={styles.modeBtnText}>{icon}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  scroll: { paddingBottom: 120 }, // room for the floating chat FAB
  headerRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  logo: { fontSize: 30, fontWeight: '800', color: COLORS.text, letterSpacing: 0.5 },
  tagline: { fontSize: 11, letterSpacing: 3, color: COLORS.faint, marginTop: 4, fontWeight: '600' },
  modeRow: { flexDirection: 'row' },
  modeBtn: {
    width: 48, height: 48, borderRadius: 24, alignItems: 'center', justifyContent: 'center',
    marginLeft: 8, backgroundColor: COLORS.card, borderWidth: 1, borderColor: COLORS.border,
  },
  modeBtnActive: { borderColor: COLORS.good, backgroundColor: COLORS.cardAlt },
  modeBtnText: { fontSize: 20 },
  badge: {
    flexDirection: 'row', alignItems: 'center', alignSelf: 'flex-start',
    borderWidth: 1, borderColor: COLORS.border, borderRadius: 999,
    paddingHorizontal: 14, paddingVertical: 8, marginTop: 18, backgroundColor: COLORS.card,
  },
  dot: { width: 8, height: 8, borderRadius: 4, marginRight: 8 },
  badgeText: { fontSize: 11, letterSpacing: 2, color: COLORS.muted, fontWeight: '700' },

  statsGrid: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between', marginTop: 20 },
  stat: {
    backgroundColor: COLORS.card, borderWidth: 1, borderColor: COLORS.border,
    borderRadius: 18, paddingVertical: 20, alignItems: 'center', marginBottom: 12,
  },
  statValue: { fontSize: 30, fontWeight: '800', color: COLORS.text },
  statLabel: { fontSize: 9, letterSpacing: 1.2, color: COLORS.faint, marginTop: 8, textAlign: 'center' },

  sectionTitle: {
    fontSize: 11, fontWeight: '700', color: COLORS.muted, letterSpacing: 2,
    marginTop: 16, marginBottom: 12,
  },
  grid: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between' },
  tile: {
    backgroundColor: COLORS.card, borderWidth: 1, borderColor: COLORS.border,
    borderRadius: 18, paddingVertical: 22, alignItems: 'center', marginBottom: 12,
    minHeight: 120, justifyContent: 'center', overflow: 'hidden',
  },
  tileIconBox: {
    width: 64, height: 64, borderRadius: 16, alignItems: 'center', justifyContent: 'center',
    backgroundColor: COLORS.cardAlt, borderWidth: 1, borderColor: COLORS.border, marginBottom: 10,
  },
  tileEmoji: { fontSize: 30 },
  tileLabel: { fontSize: 15, fontWeight: '700', color: COLORS.text },
  footer: { textAlign: 'center', fontSize: 11, color: COLORS.faint, marginTop: 18 },
});
