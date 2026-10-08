// Cross-platform image picking that works on BOTH native and Expo Web.
// On web, expo-image-picker's launchImageLibraryAsync can be unreliable, so we
// use a hidden <input type="file"> and normalize the result into an asset:
//   { uri, file, fileName, mimeType }   (file is present only on web)
import { Platform } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

export async function pickImage(fromCamera = false) {
  if (Platform.OS === 'web') {
    return pickImageWeb();
  }
  const perm = fromCamera
    ? await ImagePicker.requestCameraPermissionsAsync()
    : await ImagePicker.requestMediaLibraryPermissionsAsync();
  if (!perm.granted) {
    throw new Error('Permission denied. Allow camera/photos to continue.');
  }
  const opts = { quality: 0.8, allowsEditing: true };
  const res = fromCamera
    ? await ImagePicker.launchCameraAsync(opts)
    : await ImagePicker.launchImageLibraryAsync({
        ...opts,
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
      });
  if (res.canceled || !res.assets?.[0]) return null;
  const a = res.assets[0];
  return { uri: a.uri, fileName: a.fileName || 'leaf.jpg', mimeType: a.mimeType || 'image/jpeg' };
}

function pickImageWeb() {
  return new Promise((resolve) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = () => {
      const file = input.files?.[0];
      if (!file) return resolve(null);
      resolve({
        uri: URL.createObjectURL(file),
        file,                       // send the raw File to the server
        fileName: file.name || 'leaf.jpg',
        mimeType: file.type || 'image/jpeg',
      });
    };
    // Resolve null if the user cancels the dialog (best-effort).
    input.oncancel = () => resolve(null);
    input.click();
  });
}
