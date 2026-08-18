/* Edify AI — Credential storage abstraction.
 * Uses Electron safeStorage when available (main process IPC),
 * falls back to localStorage in dev/browser mode.
 */

type EdifyAPI = {
  credentials: {
    set: (key: string, value: string) => Promise<boolean>;
    get: (key: string) => Promise<string | null>;
    delete: (key: string) => Promise<boolean>;
  };
};

function getAPI(): EdifyAPI | null {
  if (typeof window !== "undefined" && (window as unknown as Record<string, unknown>).edify) {
    return (window as unknown as Record<string, unknown>).edify as EdifyAPI;
  }
  return null;
}

export async function saveCredential(key: string, value: string): Promise<boolean> {
  const api = getAPI();
  if (api) {
    return api.credentials.set(key, value);
  }
  // Dev fallback: localStorage (NOT for production)
  localStorage.setItem(`edify_cred_${key}`, value);
  return true;
}

export async function loadCredential(key: string): Promise<string | null> {
  const api = getAPI();
  if (api) {
    return api.credentials.get(key);
  }
  return localStorage.getItem(`edify_cred_${key}`);
}

export async function deleteCredential(key: string): Promise<boolean> {
  const api = getAPI();
  if (api) {
    return api.credentials.delete(key);
  }
  localStorage.removeItem(`edify_cred_${key}`);
  return true;
}
