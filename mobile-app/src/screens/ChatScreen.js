import React, { useRef, useState } from 'react';
import {
  FlatList,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import * as Speech from 'expo-speech';
import { COLORS, RIPPLE } from '../components/ui';
import { SafeScreen } from '../components/AndroidChrome';
import { sendChat } from '../api/client';

let idSeq = 0;
const nextId = () => `m${++idSeq}`;

export default function ChatScreen() {
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      role: 'assistant',
      content:
        'Namaste! I am the AgriSense farm assistant. Ask me about crops, soil, weather, yield, pests or government schemes.',
    },
  ]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [speakReplies, setSpeakReplies] = useState(true);
  const listRef = useRef(null);

  // Build the history array the backend expects: [{role, content}, ...]
  const toHistory = (msgs) =>
    msgs.map((m) => ({ role: m.role, content: m.content }));

  async function send(text) {
    const content = (text ?? input).trim();
    if (!content || sending) return;
    setInput('');
    Speech.stop();

    const userMsg = { id: nextId(), role: 'user', content };
    const history = toHistory([...messages, userMsg]);
    setMessages((m) => [...m, userMsg]);
    setSending(true);
    try {
      const data = await sendChat(history);
      const reply = data.response || data.reply || '…';
      setMessages((m) => [...m, { id: nextId(), role: 'assistant', content: reply }]);
      if (speakReplies) Speech.speak(reply, { language: 'en-IN' });
    } catch (e) {
      const msg = e.response?.data?.error || e.message || 'Could not reach the assistant.';
      setMessages((m) => [...m, { id: nextId(), role: 'assistant', content: `⚠ ${msg}` }]);
    } finally {
      setSending(false);
      setTimeout(() => listRef.current?.scrollToEnd({ animated: true }), 100);
    }
  }

  // Mic: on stock Expo Go we can't run a native STT module, so the mic
  // focuses the input and lets the user use the device's built-in
  // voice keyboard (dictation). See src/stt.js for the dev-build upgrade.
  function onMicPress() {
    setInputHint();
  }
  function setInputHint() {
    setInput((cur) => cur); // no-op; the TextInput itself handles dictation
    inputRef.current?.focus();
  }
  const inputRef = useRef(null);

  function renderItem({ item }) {
    const mine = item.role === 'user';
    return (
      <View style={[styles.bubble, mine ? styles.mine : styles.theirs]}>
        <Text style={[styles.bubbleText, mine && { color: COLORS.accentText }]}>{item.content}</Text>
      </View>
    );
  }

  return (
    <SafeScreen>
    <KeyboardAvoidingView
      style={styles.flex}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      keyboardVerticalOffset={90}
    >
      <FlatList
        ref={listRef}
        data={messages}
        keyExtractor={(m) => m.id}
        renderItem={renderItem}
        contentContainerStyle={styles.list}
        onContentSizeChange={() => listRef.current?.scrollToEnd({ animated: true })}
      />

      <View style={styles.toolbar}>
        <Pressable onPress={() => setSpeakReplies((s) => !s)} style={styles.toolBtn} android_ripple={{ ...RIPPLE, borderless: true }} accessibilityRole="button" accessibilityLabel="Toggle read-aloud">
          <Text style={styles.toolIcon}>{speakReplies ? '🔊' : '🔇'}</Text>
        </Pressable>
        <Text style={styles.toolLabel}>{speakReplies ? 'Read-aloud on' : 'Read-aloud off'}</Text>
      </View>

      <View style={styles.inputRow}>
        <Pressable onPress={onMicPress} style={styles.micBtn} disabled={sending} android_ripple={{ ...RIPPLE, borderless: true }} accessibilityRole="button" accessibilityLabel="Voice input">
          <Text style={styles.micIcon}>🎤</Text>
        </Pressable>
        <TextInput
          ref={inputRef}
          style={styles.input}
          placeholder="Ask about crops, soil, weather…"
          placeholderTextColor={COLORS.muted}
          value={input}
          onChangeText={setInput}
          multiline
          // Enables device voice dictation from the on-screen keyboard.
          enablesReturnKeyAutomatically
        />
        <Pressable
          onPress={() => send()}
          style={[styles.sendBtn, (sending || !input.trim()) && styles.sendDisabled]}
          disabled={sending || !input.trim()}
          android_ripple={{ ...RIPPLE, borderless: true }}
          accessibilityRole="button"
          accessibilityLabel="Send message"
        >
          <Text style={styles.sendIcon}>{sending ? '…' : '➤'}</Text>
        </Pressable>
      </View>
    </KeyboardAvoidingView>
    </SafeScreen>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: COLORS.bg },
  list: { padding: 14, paddingBottom: 8 },
  bubble: {
    maxWidth: '82%',
    borderRadius: 18,
    paddingHorizontal: 14,
    paddingVertical: 11,
    marginBottom: 10,
  },
  mine: { alignSelf: 'flex-end', backgroundColor: COLORS.accent, borderBottomRightRadius: 4 },
  theirs: { alignSelf: 'flex-start', backgroundColor: COLORS.card, borderWidth: 1, borderColor: COLORS.border, borderBottomLeftRadius: 4 },
  bubbleText: { fontSize: 15, color: COLORS.text, lineHeight: 21 },
  toolbar: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 14, paddingVertical: 4 },
  toolBtn: { padding: 4 },
  toolIcon: { fontSize: 18 },
  toolLabel: { fontSize: 12, color: COLORS.muted, marginLeft: 6 },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    padding: 12,
    backgroundColor: COLORS.card,
    borderTopWidth: 1,
    borderTopColor: COLORS.border,
  },
  micBtn: {
    width: 48, height: 48, borderRadius: 24,
    backgroundColor: COLORS.cardAlt, borderWidth: 1, borderColor: COLORS.border,
    alignItems: 'center', justifyContent: 'center', marginRight: 8,
  },
  micIcon: { fontSize: 18 },
  input: {
    flex: 1,
    maxHeight: 120,
    backgroundColor: COLORS.cardAlt,
    borderWidth: 1, borderColor: COLORS.border,
    borderRadius: 22,
    paddingHorizontal: 16,
    paddingVertical: 11,
    fontSize: 15,
    color: COLORS.text,
  },
  sendBtn: {
    width: 48, height: 48, borderRadius: 24,
    backgroundColor: COLORS.accent, alignItems: 'center', justifyContent: 'center', marginLeft: 8,
  },
  sendDisabled: { opacity: 0.5 },
  sendIcon: { color: COLORS.accentText, fontSize: 18, fontWeight: '700' },
});
