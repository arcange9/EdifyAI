import "@testing-library/jest-dom";

// Polyfill for jsdom
if (typeof globalThis.matchMedia === "undefined") {
  globalThis.matchMedia = (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
  });
}

// Polyfill IndexedDB
if (typeof globalThis.indexedDB === "undefined") {
  (globalThis as Record<string, unknown>).indexedDB = {
    open: () => ({
      onupgradeneeded: null,
      onsuccess: null,
      onerror: null,
      result: {
        objectStoreNames: { contains: () => false },
        createObjectStore: () => ({}),
        transaction: () => ({ objectStore: () => ({ put: () => ({}), get: () => ({}), getAll: () => ({}), delete: () => ({}), clear: () => ({}) }) }),
      },
    }),
  };
}
