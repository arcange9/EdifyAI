/* Edify AI — App Context
 * Central state management: loads database, builds active AI provider,
 * manages preferences and provider configs. Exposes to all pages.
 */

import {
  createContext, useCallback, useContext, useEffect, useMemo, useState,
  type ReactNode,
} from "react";
import { db } from "./db";
import { saveCredential, loadCredential, deleteCredential } from "./credentials";
import { createProvider } from "../ai/providers/manager";
import { ResilientProvider } from "../ai/providers/resilient";
import type { AIProvider } from "../ai/providers/interface";
import type { ProviderConfig, AppPreferences, Project } from "./types";

const DEFAULT_PREFS: AppPreferences = {
  theme: "system",
  onboarded: false,
  fallbackEnabled: false,
  language: "English",
};

interface AppContextValue {
  ready: boolean;
  preferences: AppPreferences;
  providers: ProviderConfig[];
  activeProvider: AIProvider | null;
  projects: Project[];
  refreshKey: number;
  // Preferences
  setPreferences: (prefs: Partial<AppPreferences>) => Promise<void>;
  // Providers
  loadProviderConfigs: () => Promise<ProviderConfig[]>;
  saveProviderConfig: (config: ProviderConfig, apiKey?: string) => Promise<void>;
  deleteProviderConfig: (id: string) => Promise<void>;
  setActiveProvider: (id: string) => Promise<void>;
  // Projects
  refreshProjects: () => Promise<void>;
  // Theme
  applyTheme: () => void;
}

const AppCtx = createContext<AppContextValue | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);
  const [preferences, setPrefs] = useState<AppPreferences>(DEFAULT_PREFS);
  const [providers, setProviders] = useState<ProviderConfig[]>([]);
  const [activeProvider, setActiveProviderState] = useState<AIProvider | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [refreshKey, setRefreshKey] = useState(0);

  const applyTheme = useCallback(() => {
    const theme = preferences.theme;
    const resolved = theme === "system"
      ? (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")
      : theme;
    document.documentElement.setAttribute("data-theme", resolved);
  }, [preferences.theme]);

  const setPreferences = useCallback(async (prefs: Partial<AppPreferences>) => {
    const updated = { ...preferences, ...prefs };
    setPrefs(updated);
    await db.savePreferences(updated);
  }, [preferences]);

  const loadProviderConfigs = useCallback(async () => {
    const configs = await db.getProviders();
    // Load API keys from secure storage
    const withKeys: ProviderConfig[] = [];
    for (const config of configs) {
      const key = await loadCredential(`provider_${config.id}`);
      withKeys.push({ ...config, apiKey: key ?? "" });
    }
    setProviders(withKeys);
    return withKeys;
  }, []);

  const saveProviderConfig = useCallback(async (config: ProviderConfig, apiKey?: string) => {
    await db.saveProvider(config);
    if (apiKey !== undefined) {
      await saveCredential(`provider_${config.id}`, apiKey);
    }
    await loadProviderConfigs();
  }, [loadProviderConfigs]);

  const deleteProviderConfig = useCallback(async (id: string) => {
    await db.deleteProvider(id);
    await deleteCredential(`provider_${id}`);
    await loadProviderConfigs();
  }, [loadProviderConfigs]);

  const setActiveProvider = useCallback(async (id: string) => {
    const config = providers.find((p) => p.id === id);
    if (!config) return;
    const provider = await createProvider(config);
    if (provider) {
      setActiveProviderState(new ResilientProvider(provider));
    }
    await setPreferences({ defaultProviderId: id });
  }, [providers, setPreferences]);

  const refreshProjects = useCallback(async () => {
    const p = await db.getProjects();
    setProjects(p.sort((a, b) => b.updatedAt - a.updatedAt));
    setRefreshKey((k) => k + 1);
  }, []);

  // Initialize
  useEffect(() => {
    (async () => {
      await db.init();
      const prefs = await db.getPreferences();
      setPrefs(prefs);
      const configs = await loadProviderConfigs();
      // Build active provider from default
      const defaultId = prefs.defaultProviderId;
      if (defaultId) {
        const config = configs.find((c) => c.id === defaultId && c.enabled);
        if (config) {
          const provider = await createProvider(config);
          if (provider) setActiveProviderState(new ResilientProvider(provider));
        }
      }
      await refreshProjects();
      setReady(true);
    })();

    // System theme listener
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = () => { if (preferences.theme === "system") applyTheme(); };
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Apply theme on preference change
  useEffect(() => {
    applyTheme();
  }, [applyTheme]);

  const value = useMemo<AppContextValue>(() => ({
    ready,
    preferences,
    providers,
    activeProvider,
    projects,
    refreshKey,
    setPreferences,
    loadProviderConfigs,
    saveProviderConfig,
    deleteProviderConfig,
    setActiveProvider,
    refreshProjects,
    applyTheme,
  }), [ready, preferences, providers, activeProvider, projects, refreshKey, setPreferences, loadProviderConfigs, saveProviderConfig, deleteProviderConfig, setActiveProvider, refreshProjects, applyTheme]);

  return <AppCtx.Provider value={value}>{children}</AppCtx.Provider>;
}

export function useApp(): AppContextValue {
  const ctx = useContext(AppCtx);
  if (!ctx) throw new Error("useApp must be used inside <AppProvider>");
  return ctx;
}
