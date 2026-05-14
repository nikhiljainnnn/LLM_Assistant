/**
 * src/components/StatusBar.tsx
 */

import { useEffect, useState } from "react";
import { getHealth } from "../lib/api";

export function StatusBar() {
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    const check = async () => {
      try { await getHealth(); setOnline(true); }
      catch { setOnline(false); }
    };
    check();
    const id = setInterval(check, 30_000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="status-bar">
      <span className={`status-dot ${online === true ? "status-dot--on" : online === false ? "status-dot--off" : "status-dot--checking"}`} />
      <span className="status-text">
        {online === true ? "Backend connected" : online === false ? "Backend offline" : "Checking…"}
      </span>
    </div>
  );
}
