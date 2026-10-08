// Small shared UI kit: consistent buttons, inputs, loading & error states
// used across all five screens.
import React from 'react';
import {
  ActivityIndicator,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

// Android ripple used on all pressables (web falls back to a subtle opacity).
export const RIPPLE = { color: 'rgba(255,255,255,0.14)', borderless: false };

export const COLORS = {
  // Dark "smart climate" theme (matches reference design)
  bg: '#07090c',          // near-black page
  card: '#0e1217',        // panel
  cardAlt: '#12161c',
  text: '#eef1f4',        // off-white
  muted: '#9aa3ab',       // grey text
  faint: '#6b7480',
  border: '#1e242c',      // hairline borders
  accent: '#e9edf1',      // primary (light) — pill buttons
  accentText: '#11151a',  // text on accent
  good: '#3ecf8e',
  error: '#ff6b6b',
  glow: '#2a3240',
};

export function Screen({ children }) {
  return <View style={styles.screen}>{children}</View>;
}

export function Title({ children }) {
  return <Text style={styles.title}>{children}</Text>;
}

export function Card({ children, style }) {
  return <View style={[styles.card, style]}>{children}</View>;
}

export function Field({ label, ...props }) {
  return (
    <View style={styles.fieldWrap}>
      <Text style={styles.label}>{label}</Text>
      <TextInput
        style={styles.input}
        placeholderTextColor={COLORS.muted}
        keyboardType="numeric"
        {...props}
      />
    </View>
  );
}

export function Button({ title, onPress, disabled, loading, variant = 'primary' }) {
  const primary = variant === 'primary';
  const bg = primary ? COLORS.accent : 'transparent';
  const fg = primary ? COLORS.accentText : COLORS.text;
  return (
    <Pressable
      style={({ pressed }) => [
        styles.button,
        { backgroundColor: bg, borderColor: primary ? COLORS.accent : COLORS.border },
        (disabled || loading) && styles.buttonDisabled,
        Platform.OS === 'web' && pressed && { opacity: 0.85 },
      ]}
      onPress={onPress}
      disabled={disabled || loading}
      android_ripple={RIPPLE}
      accessibilityRole="button"
    >
      {loading ? (
        <ActivityIndicator color={fg} />
      ) : (
        <Text style={[styles.buttonText, { color: fg }]}>{title}</Text>
      )}
    </Pressable>
  );
}

export function ErrorBox({ message }) {
  if (!message) return null;
  return (
    <View style={styles.errorBox}>
      <Text style={styles.errorText}>⚠ {message}</Text>
    </View>
  );
}

export function ResultRow({ label, value, strong }) {
  return (
    <View style={styles.resultRow}>
      <Text style={styles.resultLabel}>{label}</Text>
      <Text style={[styles.resultValue, strong && styles.resultStrong]}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: COLORS.bg, padding: 16 }, // 8dp grid
  title: {
    fontSize: 26, fontWeight: '800', color: COLORS.text, marginBottom: 4, letterSpacing: 0.3,
  },
  card: {
    backgroundColor: COLORS.card,
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  fieldWrap: { marginBottom: 12 },
  label: {
    fontSize: 11, color: COLORS.muted, marginBottom: 6, fontWeight: '700',
    textTransform: 'uppercase', letterSpacing: 1.2,
  },
  input: {
    backgroundColor: COLORS.cardAlt,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 12,
    paddingHorizontal: 14,
    paddingVertical: 12,
    fontSize: 16,
    color: COLORS.text,
  },
  button: {
    borderRadius: 14,
    minHeight: 48, // Material min touch target
    paddingVertical: 12,
    paddingHorizontal: 16,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    marginTop: 8,
    overflow: 'hidden', // keep ripple inside the rounded corners
  },
  buttonDisabled: { opacity: 0.55 },
  buttonText: { fontSize: 16, fontWeight: '700', letterSpacing: 0.2 },
  errorBox: {
    backgroundColor: '#2a1214',
    borderColor: '#5b2226',
    borderWidth: 1,
    borderRadius: 12,
    padding: 12,
    marginBottom: 12,
  },
  errorText: { color: COLORS.error, fontSize: 14 },
  resultRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: COLORS.border,
  },
  resultLabel: { fontSize: 14, color: COLORS.muted },
  resultValue: { fontSize: 16, color: COLORS.text, fontWeight: '600' },
  resultStrong: { color: COLORS.good, fontSize: 18, fontWeight: '800' },
});
