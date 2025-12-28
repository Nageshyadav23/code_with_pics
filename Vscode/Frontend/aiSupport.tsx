import React, { useEffect, useState, useRef } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  Keyboard,
  Animated,
  Platform,
} from "react-native";
import { useLocalSearchParams } from "expo-router";
import { safelyDecodeURIComponent } from "expo-router/build/fork/getStateFromPath-forks";

export default function AISupport() {
  const { code: encodedCode } = useLocalSearchParams<{ code: string }>();
  const code = safelyDecodeURIComponent(encodedCode || "");

  const [messages, setMessages] = useState<{ role: string; content: string }[]>([]);
  const [input, setInput] = useState(code || "");

  const scrollRef = useRef<ScrollView>(null);
  const inputAnim = useRef(new Animated.Value(0)).current; // animation value 0=bottom, 1=middle

  useEffect(() => {
    if (code) setInput(code);
  }, [code]);

  useEffect(() => {
    const showSub = Keyboard.addListener("keyboardDidShow", () => moveInput("up"));
    const hideSub = Keyboard.addListener("keyboardDidHide", () => moveInput("down"));
    return () => {
      showSub.remove();
      hideSub.remove();
    };
  }, []);

  const moveInput = (direction: "up" | "down") => {
    Animated.timing(inputAnim, {
      toValue: direction === "up" ? 1 : 0,
      duration: 300,
      useNativeDriver: false,
    }).start();
  };

  const sendMessage = async () => {
    if (!input.trim()) return;
    const newMessages = [...messages, { role: "user", content: input }];
    setMessages(newMessages);
    setInput("");

    try {
      const res = await fetch("http://YOUR_INTERNET_IP_ADDRESS/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: input,
          history: messages.map((m) => [m.content]),
        }),
      });
      const data = await res.json();
      console.log(data.reply);
      setMessages([...newMessages, { role: "assistant", content: data.reply }]);
      scrollRef.current?.scrollToEnd({ animated: true });
    } catch (err) {
      setMessages([...newMessages, { role: "assistant", content: "Error: " + err }]);
    }
  };

  const handleDumpCode = () => {
    if (code) setInput((prev) => (prev ? prev + "\n" + code : code));
  };

  // Animate translateY (vertical position) + opacity
  const translateY = inputAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [0, -250], // move up about mid-screen
  });
  const opacity = inputAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [1, 0.95],
  });

  return (
    <View style={styles.container}>
      <View style={styles.topButtons}>
        <TouchableOpacity style={styles.aiButton} onPress={handleDumpCode}>
          <Text style={styles.aiButtonText}>🎯</Text>
        </TouchableOpacity>
      </View>

      <Text style={styles.header}>AI Support</Text>

      <ScrollView
        ref={scrollRef}
        style={styles.chatBox}
        contentContainerStyle={{ paddingVertical: 10 }}
        onContentSizeChange={() => scrollRef.current?.scrollToEnd({ animated: true })}
        keyboardShouldPersistTaps="handled"
      >
        {messages.map((m, i) => (
          <View
            key={i}
            style={[
              styles.messageContainer,
              m.role === "user" ? styles.userMessage : styles.aiMessage,
            ]}
          >
            <Text style={styles.messageText}>{m.content}</Text>
          </View>
        ))}
      </ScrollView>

      {/* Animated Input Section */}
      <Animated.View
        style={[
          styles.inputWrapper,
          {
            transform: [{ translateY }],
            opacity,
          },
        ]}
      >
        <TextInput
          style={styles.input}
          value={input}
          onChangeText={setInput}
          placeholder="Ask AI something..."
          placeholderTextColor="#888"
          multiline
          onFocus={() => moveInput("up")}
          onBlur={() => moveInput("down")}
        />
        <TouchableOpacity style={styles.sendButton} onPress={sendMessage}>
          <Text style={styles.sendButtonText}>Send</Text>
        </TouchableOpacity>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#111", justifyContent: "flex-end" },
  topButtons: { flexDirection: "row", padding: 12 },
  aiButton: {
    backgroundColor: "#4CAF50",
    borderRadius: 24,
    paddingVertical: 10,
    paddingHorizontal: 16,
  },
  aiButtonText: { color: "white", fontWeight: "700", fontSize: 16 },
  header: {
    textAlign: "center",
    fontSize: 22,
    fontWeight: "700",
    color: "white",
    marginBottom: 10,
  },
  chatBox: { flex: 1, paddingHorizontal: 12 },
  messageContainer: {
    padding: 10,
    borderRadius: 12,
    marginVertical: 6,
    maxWidth: "80%",
  },
  userMessage: {
    backgroundColor: "#4CAF50",
    alignSelf: "flex-end",
    borderTopRightRadius: 0,
  },
  aiMessage: {
    backgroundColor: "#333",
    alignSelf: "flex-start",
    borderTopLeftRadius: 0,
  },
  messageText: { color: "white", fontSize: 16 },
  inputWrapper: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    padding: 12,
    backgroundColor: "#111",
  },
  input: {
    flex: 1,
    backgroundColor: "#1e1e1e",
    color: "white",
    borderRadius: 24,
    paddingHorizontal: 16,
    paddingVertical: 12,
    fontSize: 16,
    maxHeight: 120,
  },
  sendButton: {
    marginLeft: 10,
    backgroundColor: "#4CAF50",
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 24,
    justifyContent: "center",
  },
  sendButtonText: { color: "white", fontWeight: "700" },
});


