// Android chrome helpers: safe-area screen wrapper + floating assistant FAB.
// Pure structure/behavior — reuses the existing dark theme colors, adds none.
import React from 'react';
import { Platform, Pressable, StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import { COLORS, RIPPLE } from './ui';
import { useViewMode } from './ResponsiveShell';

// Wraps a screen so content clears the status bar (top) and nav bar (bottom).
export function SafeScreen({ children, style }) {
  const insets = useSafeAreaInsets();
  return (
    <View
      style={[
        styles.safe,
        { paddingTop: insets.top, paddingBottom: Math.max(insets.bottom, 0) },
        style,
      ]}
    >
      {children}
    </View>
  );
}

// Floating Action Button (Material) — bottom-right, opens the chat assistant.
// 56dp circle, elevation + ripple, offset for the bottom nav-bar inset and the
// view-mode toggle pill so they never overlap.
export function ChatFAB() {
  const navigation = useNavigation();
  const insets = useSafeAreaInsets();
  const { isMobile, setMode } = useViewMode();

  const openAgent = () => {
    // In desktop view the RN chat screen sits behind the desktop web frame,
    // so switch back to mobile first, then open the assistant.
    if (!isMobile) setMode('mobile');
    navigation.navigate('Chat');
  };

  return (
    <View
      pointerEvents="box-none"
      style={[
        styles.fabWrap,
        { bottom: isMobile ? 16 + insets.bottom + 56 : 24 + insets.bottom },
      ]}
    >
      <Pressable
        onPress={openAgent}
        android_ripple={{ ...RIPPLE, borderless: true }}
        accessibilityRole="button"
        accessibilityLabel="Ask AgriSense assistant"
        style={({ pressed }) => [
          styles.fab,
          Platform.OS === 'web' && pressed && { opacity: 0.85 },
        ]}
      >
        <Text style={styles.fabIcon}>💬</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: COLORS.bg },
  fabWrap: {
    position: 'absolute',
    right: 16,
    zIndex: 1200, // above the desktop web frame overlay (zIndex 1)
  },
  fab: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: COLORS.cardAlt,
    borderWidth: 1,
    borderColor: COLORS.border,
    alignItems: 'center',
    justifyContent: 'center',
    // Material elevation
    ...Platform.select({
      android: { elevation: 6 },
      web: { boxShadow: '0 6px 16px rgba(0,0,0,0.45)' },
      default: {},
    }),
  },
  fabIcon: { fontSize: 24 },
});
