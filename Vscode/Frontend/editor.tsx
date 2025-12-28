import React from "react";
import { TextInput, StyleSheet, TextInputProps } from "react-native";

type CodeEditorProps = TextInputProps & {
  language?: "python" | "java" | "c";  // 👈 custom prop
};

export default function CodeEditor({ language, ...props }: CodeEditorProps) {
  return (
    <TextInput
      {...props}
      multiline
      autoCorrect={false}
      autoCapitalize="none"
      spellCheck={false}
      style={[styles.editor, props.style]}
    />
  );
}

const styles = StyleSheet.create({
  editor: {
    fontFamily: "monospace",
    fontSize: 14,
    minHeight: 200,
    padding: 12,
    backgroundColor: "#1e1e1e", // dark background like VS Code
    color: "#d4d4d4",
    borderRadius: 8,
  },
});
