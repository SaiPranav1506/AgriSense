import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';

/**
 * React Router does not scroll to `#hash` targets on navigation, so
 * links like `/#features` land on the page but never move. This hook
 * watches the location and scrolls the matching element into view once
 * it has mounted (retrying a few frames for lazily-rendered sections).
 */
export function useHashScroll() {
  const { pathname, hash } = useLocation();

  useEffect(() => {
    if (!hash) return undefined;
    const id = decodeURIComponent(hash.replace(/^#/, ''));
    if (!id) return undefined;

    let frame = 0;
    let raf = 0;
    const tick = () => {
      const el = document.getElementById(id);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        return;
      }
      if (frame++ < 30) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [pathname, hash]);
}
