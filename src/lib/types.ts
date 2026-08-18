/* Edify AI — Core domain types.
 * Shared contracts for database, AI providers, study tools, and UI.
 * IDs are UUID strings. Timestamps are epoch ms.
 */

export type ID = string;

/* ---- Source types ---- */
export type SourceKind = "pdf" | "docx" | "txt" | "md" | "url" | "youtube" | "audio" | "blank";

/* ---- AI Provider types ---- */
export type ProviderType = "openrouter" | "google" | "groq" | "custom";

export interface ProviderConfig {
  id: string;
  type: ProviderType;
  name: string;
  apiKey: string; // stored in secure storage, loaded at runtime
  baseUrl?: string;
  model: string;
  fallbackModel?: string;
  enabled: boolean;
  updatedAt?: number;
  createdAt: number;
}

export type ModelCapability =
  | "text"
  | "vision"
  | "audio"
  | "structured_output"
  | "tool_calling"
  | "streaming"
  | "embeddings";

export interface ModelInfo {
  id: string;
  name: string;
  provider: string;
  contextLength?: number;
  capabilities: ModelCapability[];
  isFree?: boolean;
  description?: string;
}

export interface ProviderHealth {
  ok: boolean;
  message: string;
  latencyMs?: number;
}

/* ---- Projects ---- */
export interface Project {
  id: ID;
  name: string;
  description?: string;
  color: string;
  createdAt: number;
  updatedAt: number;
}

/* ---- Documents ---- */
export interface Document {
  id: ID;
  projectId: ID;
  title: string;
  sourceKind: SourceKind;
  sourceText: string;
  sourceMeta?: Record<string, string | number | undefined>;
  chunks: TextChunk[];
  createdAt: number;
  updatedAt: number;
}

export interface TextChunk {
  id: string;
  text: string;
  embedding?: number[];
  metadata?: Record<string, string | number>;
}

/* ---- Notes ---- */
export interface Note {
  id: ID;
  projectId: ID;
  documentId?: ID;
  title: string;
  content: string; // Markdown
  type: "notes" | "summary" | "study_guide" | "explanation";
  createdAt: number;
  updatedAt: number;
}

/* ---- Flashcards ---- */
export interface Flashcard {
  id: ID;
  projectId: ID;
  documentId?: ID;
  front: string;
  back: string;
  topic: string;
  difficulty: "new" | "learning" | "review" | "mastered";
  due: number;
  reps: number;
  lapses: number;
  lastReview?: number;
  createdAt: number;
}

/* ---- Quizzes ---- */
export type QuizType = "mcq" | "true_false" | "short_answer";
export type QuizDifficulty = "easy" | "medium" | "hard";

export interface QuizQuestion {
  id: ID;
  projectId: ID;
  documentId?: ID;
  type: QuizType;
  topic: string;
  difficulty: QuizDifficulty;
  question: string;
  options: string[];
  correctIndex: number;
  explanation: string;
  createdAt: number;
}

export interface QuizAttempt {
  id: ID;
  projectId: ID;
  questionId: ID;
  correct: boolean;
  userAnswer?: string;
  at: number;
}

/* ---- Chat ---- */
export interface ChatMessage {
  id: ID;
  projectId: ID;
  role: "user" | "assistant" | "system";
  content: string;
  citations?: Citation[];
  at: number;
}

export interface Citation {
  documentId: ID;
  documentTitle: string;
  chunkIndex: number;
  text: string;
}

/* ---- Study Plans ---- */
export interface StudyPlan {
  id: ID;
  projectId: ID;
  subject: string;
  goal: string;
  examDate?: number;
  daysPerWeek: number;
  minutesPerDay: number;
  difficulty: QuizDifficulty;
  tasks: StudyTask[];
  createdAt: number;
}

export interface StudyTask {
  id: string;
  day: number;
  topic: string;
  objectives: string[];
  reading?: string;
  practice?: string;
  quiz?: string;
  completed: boolean;
}

/* ---- Preferences ---- */
export interface AppPreferences {
  theme: "light" | "dark" | "system";
  onboarded: boolean;
  defaultProviderId?: string;
  fallbackProviderId?: string;
  fallbackEnabled: boolean;
  language: string;
}

/* ---- Statistics ---- */
export interface StudyStats {
  totalSessions: number;
  totalDocuments: number;
  totalFlashcards: number;
  avgQuizScore: number;
  studyStreak: number;
  lastStudyDate?: number;
}
