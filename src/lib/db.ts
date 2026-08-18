/* Edify AI — Local Database (IndexedDB via idb-free wrapper)
 * Stores projects, documents, notes, flashcards, quizzes, chats, study plans,
 * and preferences. All data is local — nothing leaves the user's machine.
 */

import type {
  Project, Document, Note, Flashcard, QuizQuestion,
  ChatMessage, StudyPlan, AppPreferences, ProviderConfig
} from "./types";

const DB_NAME = "edifyai";
const DB_VERSION = 1;

const STORES = [
  "projects", "documents", "notes", "flashcards", "quizzes",
  "quizAttempts", "chats", "studyPlans", "preferences", "providers",
] as const;

export type StoreName = typeof STORES[number];

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      for (const store of STORES) {
        if (!db.objectStoreNames.contains(store)) {
          db.createObjectStore(store, { keyPath: "id" });
        }
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export class Database {
  private db: IDBDatabase | null = null;

  async init(): Promise<void> {
    this.db = await openDB();
  }

  private async tx<T>(store: StoreName, mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
    if (!this.db) await this.init();
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(store, mode);
      const request = fn(transaction.objectStore(store));
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async put<T extends { id: string }>(store: StoreName, value: T): Promise<void> {
    await this.tx(store, "readwrite", (s) => s.put(value));
  }

  async get<T>(store: StoreName, id: string): Promise<T | undefined> {
    return this.tx(store, "readonly", (s) => s.get(id));
  }

  async getAll<T>(store: StoreName): Promise<T[]> {
    return this.tx(store, "readonly", (s) => s.getAll());
  }

  async delete(store: StoreName, id: string): Promise<void> {
    await this.tx(store, "readwrite", (s) => s.delete(id));
  }

  async clear(store: StoreName): Promise<void> {
    await this.tx(store, "readwrite", (s) => s.clear());
  }

  async queryBy<T extends { projectId?: string }>(store: StoreName, projectId: string): Promise<T[]> {
    const all = await this.getAll<T>(store);
    return all.filter((item) => item.projectId === projectId);
  }

  // Preferences (single record, id = "app")
  async getPreferences(): Promise<AppPreferences> {
    const p = await this.get<AppPreferences & { id: string }>("preferences", "app");
    return p ?? { theme: "system", onboarded: false, fallbackEnabled: false, language: "English" };
  }

  async savePreferences(prefs: AppPreferences): Promise<void> {
    await this.put("preferences", { id: "app", ...prefs });
  }

  // Provider configs
  async getProviders(): Promise<ProviderConfig[]> {
    return this.getAll<ProviderConfig>("providers");
  }

  async saveProvider(config: ProviderConfig): Promise<void> {
    await this.put("providers", config);
  }

  async deleteProvider(id: string): Promise<void> {
    await this.delete("providers", id);
  }

  // Projects
  async getProjects(): Promise<Project[]> {
    return this.getAll<Project>("projects");
  }

  // Documents by project
  async getDocuments(projectId: string): Promise<Document[]> {
    return this.queryBy<Document>("documents", projectId);
  }

  // Notes by project
  async getNotes(projectId: string): Promise<Note[]> {
    return this.queryBy<Note>("notes", projectId);
  }

  // Flashcards by project
  async getFlashcards(projectId: string): Promise<Flashcard[]> {
    return this.queryBy<Flashcard>("flashcards", projectId);
  }

  // Quiz questions by project
  async getQuizQuestions(projectId: string): Promise<QuizQuestion[]> {
    return this.queryBy<QuizQuestion>("quizzes", projectId);
  }

  // Chat by project
  async getChatMessages(projectId: string): Promise<ChatMessage[]> {
    const msgs = await this.queryBy<ChatMessage>("chats", projectId);
    return msgs.sort((a, b) => a.at - b.at);
  }

  // Study plans by project
  async getStudyPlans(projectId: string): Promise<StudyPlan[]> {
    return this.queryBy<StudyPlan>("studyPlans", projectId);
  }

  // Delete everything for a project
  async deleteProject(projectId: string): Promise<void> {
    await this.delete("projects", projectId);
    for (const store of ["documents", "notes", "flashcards", "quizzes", "quizAttempts", "chats", "studyPlans"] as StoreName[]) {
      const items = await this.queryBy<{ id: string; projectId: string }>(store, projectId);
      for (const item of items) await this.delete(store, item.id);
    }
  }
}

export const db = new Database();
