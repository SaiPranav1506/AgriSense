import { Link, useLocation } from 'react-router-dom';

/**
 * A <Link> that also works for same-page hash targets.
 *
 * With React Router, clicking `/#features` while already on `/` produces no
 * location change, so nothing scrolls and the link looks dead. This handles
 * that case by scrolling directly and updating the URL hash.
 */
export default function HashLink({ to, children, onClick, ...rest }) {
  const location = useLocation();

  const [rawPath, rawHash] = String(to).split('#');
  const targetPath = rawPath || '/';
  const hash = rawHash || '';

  const handleClick = (e) => {
    onClick?.(e);
    if (e.defaultPrevented) return;

    // Same page + hash target -> scroll manually.
    if (hash && location.pathname === targetPath) {
      e.preventDefault();
      const el = document.getElementById(hash);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        window.history.replaceState(null, '', to);
      } else {
        window.location.hash = hash;
      }
    }
  };

  return (
    <Link to={to} onClick={handleClick} {...rest}>
      {children}
    </Link>
  );
}
