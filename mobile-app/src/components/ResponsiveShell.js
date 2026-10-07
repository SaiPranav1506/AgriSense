// Wraps the app with a Mobile/Desktop view mode.
//
//  - "auto"    : follow the window width (<=560px => mobile, else desktop)
//  - "mobile"  : the React-Native (Android Material) UI, phone-width frame
//  - "desktop" : on web, renders the full desktop web app (Vite/React frontend)
//                in a frame; on native it falls back to the wide RN layout.
//
// Mobile view is the native app UI. Desktop view is the marketing/desktop web
// experience at FRONTEND_URL. A floating pill (bottom-left) swaps between them.
import React, { createContext, useContext, useMemo, useState } from 'react';
import {
  Platform,
  Pressable,
  StyleSheet,
  Text,
  useWindowDimensions,
  View,
} from 'react-native';
import { COLORS, RIPPLE } from './ui';
import { FRONTEND_URL } from '../config/api';
import { navigateTo } from '../navigationRef';

const ViewModeContext = createContext({ mode: 'auto', setMode: () => {}, isMobile: true, contentWidth: 430 });

export function useViewMode() {
  return useContext(ViewModeContext);
}

// Web-only: embeds the desktop frontend. Plain DOM <iframe> works because
// React Native Web renders through react-dom.
function DesktopWebFrame() {
  // ?embed=1 tells the desktop app to hide its own chat FAB — the shell
  // renders the AgriSense agent FAB on top instead, so there's only one.
  return React.createElement('iframe', {
    src: `${FRONTEND_URL}/?embed=1`,
    title: 'AgriSense Desktop',
    style: { border: '0', width: '100%', height: '100%', display: 'block', background: '#07090c' },
  });
}

export default function ResponsiveShell({ children }) {
  const { width } = useWindowDimensions();
  const [mode, setMode] = useState('auto'); // 'auto' | 'mobile' | 'desktop'

  const autoMobile = width <= 560;
  const isMobile = mode === 'mobile' ? true : mode === 'desktop' ? false : autoMobile;
  const contentWidth = isMobile ? 430 : 980;

  // Desktop mode shows the real desktop web app (web only).
  const showDesktopWeb = Platform.OS === 'web' && !isMobile;

  const value = useMemo(
    () => ({ mode, setMode, isMobile, contentWidth }),
    [mode, isMobile, contentWidth]
  );

  return (
    <ViewModeContext.Provider value={value}>
      <View style={styles.root}>
        {/* The React-Native tree stays mounted in both modes, so navigation
            state and the agent FAB survive the switch. In desktop mode the
            web frame is overlaid on top of it. */}
        <View style={[styles.frame, { maxWidth: contentWidth, width: '100%' }]}>
          {children}
        </View>

        {showDesktopWeb && (
          <View style={styles.desktopOverlay}>
            <DesktopWebFrame />
          </View>
        )}

        {/* Agent FAB for desktop view. Rendered out here (not inside the
            navigation tree) so it paints above the web frame — the embedded
            page hides its own chat button when ?embed=1. */}
        {showDesktopWeb && (
          <View style={styles.desktopFabWrap} pointerEvents="box-none">
            <Pressable
              onPress={() => {
                setMode('mobile');
                navigateTo('Chat');
              }}
              android_ripple={{ ...RIPPLE, borderless: true }}
              accessibilityRole="button"
              accessibilityLabel="Ask AgriSense assistant"
              style={({ pressed }) => [
                styles.desktopFab,
                Platform.OS === 'web' && pressed && { opacity: 0.85 },
              ]}
            >
              <Text style={styles.desktopFabIcon}>💬</Text>
            </Pressable>
          </View>
        )}

        {/* Floating mode pill — always reachable, incl. inside desktop mode.
            Sits bottom-LEFT so it never collides with the chat FAB. */}
        <View style={styles.pillWrap} pointerEvents="box-none">
          <Pressable
            onPress={() => setMode(showDesktopWeb ? 'mobile' : 'desktop')}
            android_ripple={{ ...RIPPLE, borderless: true }}
            accessibilityRole="button"
            accessibilityLabel={showDesktopWeb ? 'Switch to mobile view' : 'Switch to desktop view'}
            style={({ pressed }) => [
              styles.pill,
              Platform.OS === 'web' && pressed && { opacity: 0.85 },
            ]}
          >
            <Text style={styles.pillText}>
              {showDesktopWeb ? '📱 Mobile' : '💻 Desktop'}
            </Text>
          </Pressable>
        </View>
      </View>
    </ViewModeContext.Provider>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#04060a', alignItems: 'center' },
  frame: {
    flex: 1,
    backgroundColor: COLORS.bg,
    ...Platform.select({
      web: { boxShadow: '0 0 40px rgba(0,0,0,0.35)' },
      default: {},
    }),
  },
  // Absolutely positioned so zIndex applies: it must sit BELOW the agent FAB
  // (zIndex 1200) which is rendered from the React-Native tree.
  desktopOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: COLORS.bg,
    zIndex: 1,
  },
  desktopFabWrap: {
    position: 'absolute',
    right: 16,
    bottom: 24,
    zIndex: 1400, // above desktopOverlay (1)
  },
  desktopFab: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: COLORS.cardAlt,
    borderWidth: 1,
    borderColor: COLORS.border,
    alignItems: 'center',
    justifyContent: 'center',
    ...Platform.select({
      android: { elevation: 6 },
      web: { boxShadow: '0 6px 16px rgba(0,0,0,0.45)' },
      default: {},
    }),
  },
  desktopFabIcon: { fontSize: 24 },
  pillWrap: { position: 'absolute', bottom: 16, left: 16, zIndex: 1000 },
  pill: {
    flexDirection: 'row',
    alignItems: 'center',
    minHeight: 44,
    paddingHorizontal: 16,
    borderRadius: 999,
    backgroundColor: 'rgba(20,30,24,0.92)',
    borderWidth: 1,
    borderColor: COLORS.border,
    ...Platform.select({
      android: { elevation: 6 },
      web: { boxShadow: '0 4px 14px rgba(0,0,0,0.45)' },
      default: {},
    }),
  },
  pillText: { color: COLORS.text, fontSize: 13, fontWeight: '700' },
});
