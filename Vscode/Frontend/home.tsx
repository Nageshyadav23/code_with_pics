

import * as ImagePicker from 'expo-image-picker';
import { Link, useRouter } from 'expo-router';
import React, { useState } from 'react';
import { Alert, Image, Text, TouchableOpacity, View } from 'react-native';


export default function Home() {
  const router = useRouter();
  const [image, setImage] = useState<string | null>(null);

const sendImageForProcessing = async (uri: string) => {
  const formData = new FormData();
  formData.append("image", {
    uri,
    name: "photo.jpg",
    type: "image/jpeg",
  } as any);

  try {
    const response = await fetch("https://GOOGLECOLAB_MAIN_PROGRAM_GENERATED_LINK/process", {
      method: "POST",
      body: formData,
    });

    const text = await response.text();
    // console.log("Raw server response:", text);

    let result;
    try {
      result = JSON.parse(text);
    } catch (err) {
      console.error("JSON parse failed:", err);
      return;
    }

const { rawText,cleanedText,mappedText, error } = result;
console.log(result)
const mapdata=encodeURIComponent(mappedText)
const rawdata=encodeURIComponent(rawText)

console.log("Raw:", rawText);
// console.log("clean",cleanedText)
// console.log(typeof(rawText))
// console.log("Cleaned:", cleanedCode);

if (rawText) {
router.push({
  pathname: "/codeEditor",
  params: {
    readrawcode: rawdata,
    readmapcode:mapdata // encode safely
  }
});


} else {
  Alert.alert("Processing failed", error || "No code extracted");
}


  } catch (error) {
    console.error("Error uploading:", error);
    Alert.alert("Upload failed", "Could not process image.");
  }
};




  const openCamera = async () => {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Permission needed', 'Camera access is required');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      const uri = result.assets[0].uri;
      setImage(uri);
      console.log('Camera Image URI:', uri);
      await sendImageForProcessing(uri); // 🚀 send to backend
    }
  };

  const openGallery = async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Permission needed', 'Gallery access is required');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      const uri = result.assets[0].uri;
      setImage(uri);
      console.log('Gallery Image URI:', uri);
      await sendImageForProcessing(uri); // 🚀 send to backend
    }
  };

  return (
    <View style={{
      display: "flex",
      gap: 60,
      justifyContent: "center",
      alignItems: "center",
      width: 380,
      height: 700,
      backgroundColor: "black"
    }}>
      <View style={{ display: "flex", flexDirection: "row", gap: 30 }}>
        <TouchableOpacity onPress={openCamera}>
          <Image source={require("../../assets/images/camera.png")} style={{ width: 100, height: 100 }} />
        </TouchableOpacity>
        <TouchableOpacity onPress={openGallery}>
          <Image source={require("../../assets/images/gallery.png")} style={{ width: 100, height: 100 }} />
        </TouchableOpacity>
      </View>

      <TouchableOpacity style={{
        backgroundColor: "white",
        paddingVertical: 10,
        paddingHorizontal: 30,
        borderRadius: 30
      }}>
        <Link href='/trainMode'>
          <Text style={{ color: "black", fontSize: 20 }}>Train Mode</Text>
        </Link>
      </TouchableOpacity>
    </View>
  )
}

