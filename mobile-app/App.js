import React from 'react';
import { Platform } from 'react-native';
import { NavigationContainer, DarkTheme } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { StatusBar } from 'expo-status-bar';

import HomeScreen from './src/screens/HomeScreen';
import CropScreen from './src/screens/CropScreen';
import YieldScreen from './src/screens/YieldScreen';
import DiseaseScreen from './src/screens/DiseaseScreen';
import ChatScreen from './src/screens/ChatScreen';
import ResponsiveShell from './src/components/ResponsiveShell';
import { ChatFAB } from './src/components/AndroidChrome';
import { navigationRef } from './src/navigationRef';
import { COLORS } from './src/components/ui';

const Stack = createNativeStackNavigator();

const theme = {
  ...DarkTheme,
  colors: {
    ...DarkTheme.colors,
    background: COLORS.bg,
    card: COLORS.bg,
    primary: COLORS.accent,
    text: COLORS.text,
    border: COLORS.border,
  },
};

export default function App() {
  return (
    <SafeAreaProvider>
      <ResponsiveShell>
        <NavigationContainer ref={navigationRef} theme={theme}>
          {/* Dark status bar with light icons (adapts to the dark theme). */}
          <StatusBar style="light" backgroundColor={COLORS.bg} translucent={false} />
          <Stack.Navigator
            screenOptions={{
              headerStyle: { backgroundColor: COLORS.bg },
              headerTintColor: COLORS.text,
              headerTitleStyle: { fontWeight: '800', letterSpacing: 0.5 },
              headerShadowVisible: false,
              contentStyle: { backgroundColor: COLORS.bg },
              // Android-style horizontal slide; hardware back handled by the stack.
              animation: Platform.OS === 'android' ? 'slide_from_right' : 'default',
            }}
          >
            <Stack.Screen name="Home" component={HomeScreen} options={{ headerShown: false }} />
            <Stack.Screen name="Crop" component={CropScreen} options={{ title: 'Crop Recommendation' }} />
            <Stack.Screen name="Yield" component={YieldScreen} options={{ title: 'Yield Prediction' }} />
            <Stack.Screen name="Disease" component={DiseaseScreen} options={{ title: 'Disease Detection' }} />
            <Stack.Screen name="Chat" component={ChatScreen} options={{ title: 'Ask AgriSense' }} />
          </Stack.Navigator>
          {/* Floating assistant available from any screen. */}
          <ChatFAB />
        </NavigationContainer>
      </ResponsiveShell>
    </SafeAreaProvider>
  );
}
