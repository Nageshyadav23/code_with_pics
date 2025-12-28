
import { useLocalSearchParams, useRouter } from "expo-router";
import { safelyDecodeURIComponent } from "expo-router/build/fork/getStateFromPath-forks";
import React, { useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  Animated,
  ScrollView,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import CodeEditor from "../(tabs)/editor";

type Lang = "python" | "java" | "c";

const RUN_API = "https://GOOGLECOLAB_MAIN_PROGRAM_GENERATED_LINK/run";
const UPDATE_API = "https://GOOGLECOLAB_MAIN_PROGRAM_GENERATED_LINK/update";
const METALAMMA_API = "http://YOUR_INTERNET_IP_ADDRESS/chat"; // 👈 same as used in your aiSupport screen

export default function CodeEditorScreen() {
  const { readrawcode, readmapcode } = useLocalSearchParams<{
    readrawcode: string;
    readmapcode: string;
  }>();

  const rawTextDecoded = safelyDecodeURIComponent(readrawcode || "");
  const mappedTextDecoded = safelyDecodeURIComponent(readmapcode || "");
  const router = useRouter();

  const [language, setLanguage] = useState<Lang>("python");
  const [rawTextCopy] = useState<string>(rawTextDecoded || "# Raw OCR not available");
  const [rawText, setRawText] = useState<string>(rawTextDecoded || "# Raw OCR not available");
  const [mappedText] = useState<string>(mappedTextDecoded || "# Mapped code not available");
  const [cleanedCode, setCleanedCode] = useState<string>(
    mappedTextDecoded || "# Write your code here"
  );
  const [viewMode, setViewMode] = useState<"mapped" | "raw">("mapped");
  const [stdin, setStdin] = useState<string>("");
  const [output, setOutput] = useState<string>("");
  const [running, setRunning] = useState(false);
  const [updating, setUpdating] = useState(false);
  const [filling, setFilling] = useState(false);

  const [toastMessage, setToastMessage] = useState<string>("");
  const [toastType, setToastType] = useState<"success" | "error">("success");
  const toastOpacity = useMemo(() => new Animated.Value(0), []);

  const displayedCode = viewMode === "raw" ? rawText : cleanedCode;

  useEffect(() => {
    if (viewMode === "mapped") setCleanedCode(mappedText);
  }, [mappedText]);

  const langPlaceholder = useMemo(() => {
    if (language === "python")
      return "# Python code\nprint('Hello from Python')";
    if (language === "java")
      return `// Java code\nclass Main {\n  public static void main(String[] args){\n    System.out.println("Hello from Java");\n  }\n}`;
    return `/* C code */\n#include <stdio.h>\nint main(){\n  printf("Hello from C\\n");\n  return 0;\n}`;
  }, [language]);

  const showToast = (message: string, type: "success" | "error" = "success") => {
    setToastMessage(message);
    setToastType(type);
    Animated.sequence([
      Animated.timing(toastOpacity, { toValue: 1, duration: 300, useNativeDriver: true }),
      Animated.delay(2000),
      Animated.timing(toastOpacity, { toValue: 0, duration: 300, useNativeDriver: true }),
    ]).start();
  };

  const runCode = async () => {
    try {
      setRunning(true);
      setOutput("");
      const codeToRun = viewMode === "raw" ? rawText : cleanedCode;

      const res = await fetch(RUN_API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ language, code: codeToRun, stdin }),
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      const combined =
        `> exit_code: ${data.exit_code}\n` +
        (data.stdout ? `\n[stdout]\n${data.stdout}` : "") +
        (data.stderr ? `\n[stderr]\n${data.stderr}` : "");
      setOutput(combined);
    } catch (e: any) {
      console.error(e);
      showToast(`Run failed: ${e?.message ?? "Unknown error"}`, "error");
    } finally {
      setRunning(false);
    }
  };

  const updateLLM = async () => {
    try {
      setUpdating(true);
      const res = await fetch(UPDATE_API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          rawText: rawTextCopy,
          correctedCode: rawText,
          mappedCode: mappedText,
        }),
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      showToast(data.message || "Update complete!", "success");

      if (data.changedMappings && data.changedMappings.length > 0) {
        router.push({
          pathname: "/changeScreen",
          params: { changedMappings: JSON.stringify(data.changedMappings) },
        });
      }
    } catch (e: any) {
      console.error(e);
      showToast(`Update failed: ${e?.message ?? "Unknown error"}`, "error");
    } finally {
      setUpdating(false);
    }
  };

  const switchLang = (newLang: Lang) => {
    setLanguage(newLang);
    if (cleanedCode === "# Write your code here") setCleanedCode(langPlaceholder);
  };

  const handleAISupport = () => {
    const aicode = viewMode === "raw" ? rawText : cleanedCode;
    router.push({
      pathname: "/aiSupport",
      params: { code: encodeURIComponent(aicode) },
    });
  };

  // 🧠 New "Fill" button functionality
const handleFill = async () => {
  const currentCode = viewMode === "raw" ? rawText : cleanedCode;
  if (!currentCode.trim()) {
    showToast("Code block is empty!", "error");
    return;
  }

  try {
    setFilling(true);
    const prompt = `Convert the following code into ${language}:\n\n${currentCode}`;
    const res = await fetch(METALAMMA_API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: prompt,
        history: [],
      }),
    });

    const data = await res.json();
    let reply = data.reply?.trim() || "";

    // 🧠 Extract code inside triple backticks
    const codeMatch = reply.match(/```[a-zA-Z]*\n([\s\S]*?)```/);
    let code = codeMatch ? codeMatch[1] : reply;

    // 🧹 Remove single-line comments: //, #, and multi-line /* ... */
    code = code
      .replace(/\/\/.*$/gm, "")         // remove //
      .replace(/#.*$/gm, "")            // remove #
      .replace(/\/\*[\s\S]*?\*\//g, "") // remove /* ... */
      .replace(/^\s*[\r\n]/gm, "")      // remove empty lines
      .trim();

    if (code) {
      setCleanedCode(code);
      showToast("Code converted successfully!", "success");
    } else {
      showToast("No valid code found in response.", "error");
    }
  } catch (err: any) {
    console.error(err);
    showToast("Error while converting code.", "error");
  } finally {
    setFilling(false);
  }
};



  return (
    <View style={{ flex: 1, backgroundColor: "black" }}>
      {/* Toast */}
      <Animated.View
        pointerEvents="none"
        style={{
          position: "absolute",
          top: 40,
          left: 20,
          right: 20,
          padding: 12,
          backgroundColor: toastType === "success" ? "green" : "red",
          borderRadius: 8,
          opacity: toastOpacity,
          zIndex: 10,
        }}
      >
        <Text style={{ color: "white", fontWeight: "600", textAlign: "center" }}>
          {toastMessage}
        </Text>
      </Animated.View>

      <ScrollView
        contentContainerStyle={{ padding: 12 }}
        keyboardShouldPersistTaps="handled"
      >
        {/* Language Tabs + Update LLM */}
        <View style={{ flexDirection: "row", gap: 8, marginBottom: 10 }}>
          {(["python", "java", "c"] as Lang[]).map((l) => (
            <TouchableOpacity
              key={l}
              onPress={() => switchLang(l)}
              style={{
                paddingVertical: 8,
                paddingHorizontal: 14,
                borderRadius: 20,
                backgroundColor: language === l ? "white" : "#222",
                borderWidth: 1,
                borderColor: "#444",
              }}
            >
              <Text style={{ color: language === l ? "black" : "white", fontWeight: "600" }}>
                {l.toUpperCase()}
              </Text>
            </TouchableOpacity>
          ))}

          <TouchableOpacity
            onPress={updateLLM}
            disabled={updating}
            style={{
              paddingVertical: 8,
              paddingHorizontal: 14,
              borderRadius: 20,
              backgroundColor: updating ? "green" : "red",
              borderWidth: 1,
              borderColor: "#444",
            }}
          >
            <Text style={{ color: "white", fontWeight: "600" }}>
              {updating ? "Updating..." : "Update LLM"}
            </Text>
          </TouchableOpacity>
        </View>

        {/* Toggle between Mapped & Raw */}
        <View style={{ flexDirection: "row", gap: 8, marginBottom: 12 }}>
          <TouchableOpacity
            onPress={() => setViewMode("mapped")}
            style={{
              flex: 1,
              paddingVertical: 10,
              borderRadius: 8,
              backgroundColor: viewMode === "mapped" ? "#4CAF50" : "#222",
            }}
          >
            <Text style={{ color: "white", textAlign: "center", fontWeight: "600" }}>
              Mapped Code
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            onPress={() => setViewMode("raw")}
            style={{
              flex: 1,
              paddingVertical: 10,
              borderRadius: 8,
              backgroundColor: viewMode === "raw" ? "#4CAF50" : "#222",
            }}
          >
            <Text style={{ color: "white", textAlign: "center", fontWeight: "600" }}>
              Raw OCR
            </Text>
          </TouchableOpacity>
        </View>

        {/* Scrollable Code Editor */}
        <View
          style={{
            maxHeight: 400,
            borderRadius: 8,
            borderColor: "#333",
            borderWidth: 1,
            overflow: "hidden",
            marginBottom: 16,
          }}
        >
          <ScrollView nestedScrollEnabled>
            <CodeEditor
              value={displayedCode}
              onChangeText={(text: string) => {
                if (viewMode === "mapped") setCleanedCode(text);
                else setRawText(text);
              }}
              multiline
              style={{
                backgroundColor: "#111",
                color: "white",
                fontSize: 14,
                padding: 10,
                minHeight: 250,
              }}
            />
          </ScrollView>
        </View>

        {/* Stdin */}
        <Text style={{ color: "#ccc", marginBottom: 6 }}>Program Input (stdin)</Text>
        <TextInput
          value={stdin}
          onChangeText={setStdin}
          placeholder="Optional input passed to your program..."
          placeholderTextColor="#777"
          multiline
          style={{
            minHeight: 60,
            color: "white",
            backgroundColor: "#0f0f0f",
            borderRadius: 12,
            padding: 10,
            borderWidth: 1,
            borderColor: "#333",
            marginBottom: 16,
          }}
        />

        {/* Run + Clear + AI + Fill */}
        <View style={{ flexDirection: "row", gap: 10, marginBottom: 16 }}>
          <TouchableOpacity
            onPress={runCode}
            disabled={running}
            style={{
              backgroundColor: running ? "#666" : "#4CAF50",
              paddingVertical: 12,
              paddingHorizontal: 24,
              borderRadius: 24,
            }}
          >
            {running ? (
              <ActivityIndicator color="white" />
            ) : (
              <Text style={{ color: "white", fontWeight: "700" }}>▶ Run</Text>
            )}
          </TouchableOpacity>

          <TouchableOpacity
            onPress={() => setOutput("")}
            style={{
              backgroundColor: "#E53935",
              paddingVertical: 12,
              paddingHorizontal: 24,
              borderRadius: 24,
            }}
          >
            <Text style={{ color: "white", fontWeight: "700" }}>Clear</Text>
          </TouchableOpacity>

          <TouchableOpacity
            onPress={handleAISupport}
            style={{
              backgroundColor: "#f4e809ff",
              paddingVertical: 12,
              paddingHorizontal: 20,
              borderRadius: 24,
            }}
          >
            <Text style={{ color: "black", fontWeight: "700" }}>🤖 AI</Text>
          </TouchableOpacity>

          <TouchableOpacity
            onPress={handleFill}
            disabled={filling}
            style={{
              backgroundColor: filling ? "#999" : "#00BCD4",
              paddingVertical: 12,
              paddingHorizontal: 20,
              borderRadius: 24,
            }}
          >
            <Text style={{ color: "white", fontWeight: "700" }}>
              {filling ? "Filling..." : "Fill"}
            </Text>
          </TouchableOpacity>
        </View>

        {/* Scrollable Output */}
        <Text style={{ color: "#ccc", marginBottom: 6 }}>Output</Text>
        <ScrollView
          style={{
            minHeight: 200,
            maxHeight: 300,
            backgroundColor: "#0f0f0f",
            borderRadius: 12,
            padding: 10,
            borderWidth: 1,
            borderColor: "#333",
          }}
          nestedScrollEnabled
        >
          <Text style={{ color: "white", fontFamily: "monospace" }}>
            {output || "Run your program to see output here..."}
          </Text>
        </ScrollView>
      </ScrollView>
    </View>
  );
}
























