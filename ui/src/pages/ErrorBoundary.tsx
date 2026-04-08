import React from "react";

type Props = { children: React.ReactNode };
type State = { hasError: boolean; message: string };

export default class ErrorBoundary extends React.Component<Props, State> {
  state: State = { hasError: false, message: "" };

  static getDerivedStateFromError(err: unknown): State {
    const message = err instanceof Error ? err.message : String(err);
    return { hasError: true, message };
  }

  componentDidCatch(err: unknown) {
    // optional: console log
    console.error("ErrorBoundary caught:", err);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="rounded-xl border border-rose-400/30 bg-rose-500/10 p-4 text-sm text-rose-100 break-words">
          <div className="font-semibold">Viewer crashed</div>
          <div className="mt-1 text-xs opacity-80">{this.state.message}</div>
          <div className="mt-3 text-xs text-slate-200">
            Go back and open a different run, or refresh after fixing VTK profile import.
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
