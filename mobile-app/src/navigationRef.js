// App-wide navigation ref.
//
// The desktop view draws its agent FAB outside the React-Navigation tree
// (so it can sit above the embedded web frame), which means it can't use
// the useNavigation() hook. This ref lets it navigate directly.
import { createNavigationContainerRef } from '@react-navigation/native';

export const navigationRef = createNavigationContainerRef();

export function navigateTo(name) {
  if (navigationRef.isReady()) navigationRef.navigate(name);
}
