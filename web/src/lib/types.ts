export type SourceLanguage = 'auto' | 'zh' | 'th' | 'en'
export type TargetLanguage = 'zh' | 'th' | 'en'
export type TranslationKind = 'text' | 'image' | 'document'

export interface TranslationResult {
  id: string
  kind: TranslationKind
  source_language: TargetLanguage
  target_language: TargetLanguage
  source_text: string
  translated_text: string
  provider: string
  warnings: string[]
  filename?: string | null
  export_filename?: string | null
  created_at: string
}

export interface HealthStatus {
  status: 'ok'
  provider: string
  model: string
}

export interface TranslationOptions {
  source_language: SourceLanguage
  target_language: TargetLanguage
  context: string
}

export interface DocumentTranslationJob {
  job_id: string
  status: 'processing' | 'completed' | 'failed'
  stage: string
  completed_pages: number
  total_pages: number
  progress: number
  message: string
  error?: string | null
  result?: TranslationResult | null
}

export interface KnowledgeEntry {
  id: string
  thai_text: string
  chinese_text?: string | null
  english_text?: string | null
  source_filename: string
  created_at: string
  updated_at: string
}

export interface KnowledgeList {
  total: number
  entries: KnowledgeEntry[]
}

export interface KnowledgeImportResult {
  filename: string
  total_rows: number
  inserted: number
  updated: number
  invalid_rows: number
}

export interface KnowledgeEntryInput {
  thai_text: string
  chinese_text?: string | null
  english_text?: string | null
}
