import { Component } from 'react';
import { Link } from 'react-router-dom';

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    console.error('ErrorBoundary caught:', error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center px-6">
          <div className="glass-card p-10 text-center max-w-md">
            <div className="text-5xl mb-4">⚠️</div>
            <h2 className="font-heading text-2xl font-bold text-white-soft mb-3">
              Something went wrong
            </h2>
            <p className="text-mid font-body text-sm mb-6">
              {this.state.error?.message || 'An unexpected error occurred.'}
            </p>
            <Link
              to="/"
              onClick={() => this.setState({ hasError: false, error: null })}
              className="leaf-btn inline-block px-6 py-3 font-heading text-sm"
            >
              Back to Home
            </Link>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}