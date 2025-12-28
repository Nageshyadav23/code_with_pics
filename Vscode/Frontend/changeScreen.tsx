import { useRouter } from "expo-router";
import { useLocalSearchParams } from "expo-router/build/hooks";
import React, { useState } from "react";
import { ActivityIndicator, Alert, ScrollView, Text, TouchableOpacity, View } from "react-native";

type Mapping = { label: string; line?: number; type?: string; variant: string };

export default function ChangeScreen() {
  const router = useRouter();
  const { changedMappings } = useLocalSearchParams<{ changedMappings: string }>();

  // Parse passed mappings (should be JSON string of objects)
  const parsedMappings: Mapping[] = changedMappings ? JSON.parse(changedMappings) : [];

  const [mappingsArray, setMappingsArray] = useState<Mapping[]>(parsedMappings);
  const [checkedMappings, setCheckedMappings] = useState<boolean[]>(parsedMappings.map(() => true));
  const [loading, setLoading] = useState(false);

  const toggleCheck = (index: number) => {
    const newChecked = [...checkedMappings];
    newChecked[index] = !newChecked[index];
    setCheckedMappings(newChecked);
  };

  const handleOK = async () => {
    try {
      setLoading(true);

      // Only send confirmed (checked) mappings
      const payload = mappingsArray
        .filter((_, idx) => checkedMappings[idx])
        .map(({ label, variant }) => ({ label, variant }));

      if (payload.length === 0) {
        Alert.alert("No changes selected", "Please select at least one mapping to update.");
        return;
      }

      const res = await fetch("https://GOOGLECOLAB_MAIN_PROGRAM_GENERATED_LINK/okapi", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ changes: payload }),
      });

      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();

      Alert.alert("Success", data.message || "DB updated successfully!");
      router.back(); // return to previous screen
    } catch (e: any) {
      console.error(e);
      Alert.alert("Error", e?.message ?? "Failed to call OK API");
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = () => {
    router.back(); // simply go back
  };

  return (
    <View style={{ flex: 1, backgroundColor: "black", padding: 12 }}>
      <Text style={{ color: "white", fontSize: 18, fontWeight: "600", marginBottom: 12 }}>
        Confirm Mappings
      </Text>

      <ScrollView style={{ flex: 1, marginBottom: 12 }}>
        {mappingsArray.map((mapping, idx) => (
          <TouchableOpacity
            key={idx}
            onPress={() => toggleCheck(idx)}
            style={{
              flexDirection: "row",
              alignItems: "center",
              padding: 10,
              marginBottom: 4,
              backgroundColor: checkedMappings[idx] ? "#4CAF50" : "#222",
              borderRadius: 6,
            }}
          >
            <Text style={{ color: "white", fontFamily: "monospace" }}>
              {mapping.label}{'--->'} {mapping.variant}
            </Text>
          </TouchableOpacity>
        ))}
      </ScrollView>

      {/* OK / Cancel Buttons */}
      <View style={{ flexDirection: "row", gap: 10 }}>
        <TouchableOpacity
          onPress={handleOK}
          disabled={loading}
          style={{
            flex: 1,
            backgroundColor: "#4CAF50",
            paddingVertical: 14,
            borderRadius: 24,
            alignItems: "center",
          }}
        >
          {loading ? <ActivityIndicator color="white" /> : <Text style={{ color: "white", fontWeight: "700" }}>OK</Text>}
        </TouchableOpacity>

        <TouchableOpacity
          onPress={handleCancel}
          style={{
            flex: 1,
            backgroundColor: "#E53935",
            paddingVertical: 14,
            borderRadius: 24,
            alignItems: "center",
          }}
        >
          <Text style={{ color: "white", fontWeight: "700" }}>Cancel</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}
