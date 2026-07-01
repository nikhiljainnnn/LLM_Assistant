import { useState, useEffect } from "react";
import { login, register } from "../lib/api";

export function AuthScreen({ onAuthSuccess }: { onAuthSuccess: () => void }) {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Mouse move effect for background glows
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      const x = e.clientX / window.innerWidth;
      const y = e.clientY / window.innerHeight;
      
      const cyanGlow = document.querySelector('.glow-cyan') as HTMLElement;
      const purpleGlow = document.querySelector('.glow-purple') as HTMLElement;
      
      if (cyanGlow) cyanGlow.style.transform = `translate(${x * 20}px, ${y * 20}px)`;
      if (purpleGlow) purpleGlow.style.transform = `translate(${(1 - x) * 20}px, ${(1 - y) * 20}px)`;
    };
    
    document.addEventListener('mousemove', handleMouseMove);
    return () => document.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      if (isLogin) {
        const token = await login(username, password);
        localStorage.setItem("access_token", token.access_token);
        onAuthSuccess();
      } else {
        const token = await register(username, email, password);
        localStorage.setItem("access_token", token.access_token);
        onAuthSuccess();
      }
    } catch (err: any) {
      setError(err.message || "Authentication failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen bg-background overflow-hidden flex flex-col font-body-md text-on-surface">
      {/* Background Ambience */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="glow-cyan absolute top-0 left-0 w-[800px] h-[800px] bg-primary-container/10 rounded-full blur-[120px] -translate-x-1/2 -translate-y-1/2 transition-transform duration-1000 ease-out" />
        <div className="glow-purple absolute bottom-0 right-0 w-[600px] h-[600px] bg-secondary-container/10 rounded-full blur-[100px] translate-x-1/3 translate-y-1/3 transition-transform duration-1000 ease-out" />
      </div>

      <main className="flex-grow flex items-center justify-center p-md z-10">
        <div className="w-full max-w-md glass-panel p-xl rounded-xl relative group">
          <div className="flex flex-col items-center gap-lg">
            <div className="text-center">
              <h2 className="font-headline-md text-3xl md:text-4xl text-primary mb-2 font-bold">NEXUS</h2>
              <p className="text-sm text-on-surface-variant">Access the singularity</p>
            </div>

            {/* Pills Toggle */}
            <div className="w-full bg-surface-container-low p-1 rounded-full flex gap-1">
              <button
                className={`flex-1 py-2 rounded-full text-sm transition-all duration-300 ${isLogin ? 'bg-surface-container-highest text-primary' : 'text-on-surface-variant hover:text-on-surface'}`}
                onClick={() => { setIsLogin(true); setError(null); }}
              >
                Login
              </button>
              <button
                className={`flex-1 py-2 rounded-full text-sm transition-all duration-300 ${!isLogin ? 'bg-surface-container-highest text-primary' : 'text-on-surface-variant hover:text-on-surface'}`}
                onClick={() => { setIsLogin(false); setError(null); }}
              >
                Sign Up
              </button>
            </div>
          </div>

          {error && (
            <div className="mt-4 p-3 bg-error-container/20 border border-error/50 text-error rounded-lg text-sm text-center">
              {error}
            </div>
          )}

          {/* Form */}
          <form className="flex flex-col gap-4 mt-6 transition-all duration-300" onSubmit={handleSubmit}>
            <div className="flex flex-col gap-1">
              <label className="text-xs px-1 text-on-surface-variant uppercase tracking-wider">Username</label>
              <div className="flex items-center gap-2 px-4 py-3 bg-surface-container-highest/30 border border-white/5 rounded-lg focus-within:border-primary-container transition-colors">
                <input
                  className="bg-transparent border-none outline-none w-full text-sm text-on-surface placeholder-outline/50"
                  placeholder="Enter your username"
                  type="text"
                  required
                  value={username}
                  onChange={e => setUsername(e.target.value)}
                />
              </div>
            </div>

            {!isLogin && (
              <div className="flex flex-col gap-1">
                <label className="text-xs px-1 text-on-surface-variant uppercase tracking-wider">Email</label>
                <div className="flex items-center gap-2 px-4 py-3 bg-surface-container-highest/30 border border-white/5 rounded-lg focus-within:border-primary-container transition-colors">
                  <input
                    className="bg-transparent border-none outline-none w-full text-sm text-on-surface placeholder-outline/50"
                    placeholder="email@example.com"
                    type="email"
                    required
                    value={email}
                    onChange={e => setEmail(e.target.value)}
                  />
                </div>
              </div>
            )}

            <div className="flex flex-col gap-1">
              <label className="text-xs px-1 text-on-surface-variant uppercase tracking-wider">Password</label>
              <div className="flex items-center gap-2 px-4 py-3 bg-surface-container-highest/30 border border-white/5 rounded-lg focus-within:border-primary-container transition-colors">
                <input
                  className="bg-transparent border-none outline-none w-full text-sm text-on-surface placeholder-outline/50"
                  placeholder={isLogin ? "••••••••" : "Create a strong password"}
                  type="password"
                  required
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                />
              </div>
            </div>

            <button
              className="w-full mt-2 py-3 bg-primary-container/20 border border-primary-container/50 hover:bg-primary-container hover:text-on-primary-container text-primary-container font-semibold rounded-lg transition-all duration-300 disabled:opacity-50"
              type="submit"
              disabled={loading}
            >
              {loading ? "Processing..." : (isLogin ? "Continue" : "Create Account")}
            </button>
          </form>

          {/* Footer Text */}
          <p className="text-center text-xs text-on-surface-variant mt-6">
            By continuing, you agree to our <a className="text-on-surface border-b border-on-surface/20 hover:text-primary-container" href="#">Terms</a> and <a className="text-on-surface border-b border-on-surface/20 hover:text-primary-container" href="#">Privacy</a>.
          </p>
        </div>
      </main>

      <footer className="py-6 w-full flex justify-center items-center gap-6 text-on-surface-variant text-xs">
        <p>© 2024 NEXUS AI.</p>
      </footer>
    </div>
  );
}
