import { Link } from 'react-router-dom';

export default function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center px-6 pt-20">
      <div className="glass-card p-12 text-center max-w-md">
        <div className="text-6xl mb-4">🌾</div>
        <h1 className="font-heading text-5xl font-bold text-white-soft mb-3">404</h1>
        <p className="font-heading text-xl text-chrome mb-4">Page not found</p>
        <p className="text-mid font-body text-sm mb-8">
          This field doesn&apos;t seem to be in our farm map.
        </p>
        <Link
          to="/"
          className="leaf-btn inline-block px-6 py-3 font-heading text-base"
        >
          ← Back to Home
        </Link>
      </div>
    </div>
  );
}