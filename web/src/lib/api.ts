import type {
  DocumentTranslationJob,
  HealthStatus,
  KnowledgeImportResult,
  KnowledgeEntry,
  KnowledgeEntryInput,
  KnowledgeList,
  TranslationKind,
  TranslationOptions,
  TranslationResult,
} from './types'

const apiBase = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBase}${path}`, init)
  if (!response.ok) {
    let message = `请求失败 (${response.status})`
    try {
      const data = await response.json()
      if (typeof data.detail === 'string') message = data.detail
    } catch {
      // Keep the status-based fallback when a gateway returns a non-JSON error.
    }
    throw new Error(message)
  }
  return response.json() as Promise<T>
}

export const api = {
  health(): Promise<HealthStatus> {
    return request('/api/health')
  },

  translateText(text: string, options: TranslationOptions): Promise<TranslationResult> {
    return request('/api/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, ...options }),
    })
  },

  translateFile(
    kind: Exclude<TranslationKind, 'text'>,
    file: File,
    options: TranslationOptions,
  ): Promise<TranslationResult> {
    const form = new FormData()
    form.append('file', file)
    form.append('source_language', options.source_language)
    form.append('target_language', options.target_language)
    form.append('context', options.context)
    return request(`/api/translate/${kind}`, { method: 'POST', body: form })
  },

  createDocumentJob(
    file: File,
    options: TranslationOptions,
    signal?: AbortSignal,
  ): Promise<DocumentTranslationJob> {
    const form = new FormData()
    form.append('file', file)
    form.append('source_language', options.source_language)
    form.append('target_language', options.target_language)
    form.append('context', options.context)
    return request('/api/translate/document/jobs', {
      method: 'POST',
      body: form,
      signal,
    })
  },

  getDocumentJob(jobId: string, signal?: AbortSignal): Promise<DocumentTranslationJob> {
    return request(`/api/translate/document/jobs/${encodeURIComponent(jobId)}`, { signal })
  },

  listHistory(): Promise<TranslationResult[]> {
    return request('/api/history')
  },

  exportUrl(itemId: string, inline = false): string {
    const suffix = inline ? '?inline=true' : ''
    return `${apiBase}/api/history/${encodeURIComponent(itemId)}/export${suffix}`
  },

  textExportUrl(itemId: string): string {
    return `${apiBase}/api/history/${encodeURIComponent(itemId)}/export/text`
  },

  unformattedExportUrl(itemId: string, inline = false): string {
    const suffix = inline ? '?inline=true' : ''
    return `${apiBase}/api/history/${encodeURIComponent(itemId)}/export/unformatted${suffix}`
  },

  knowledgeTemplateUrl(): string {
    return `${apiBase}/api/knowledge/template`
  },

  listKnowledge(): Promise<KnowledgeList> {
    return request('/api/knowledge')
  },

  importKnowledge(file: File): Promise<KnowledgeImportResult> {
    const form = new FormData()
    form.append('file', file)
    return request('/api/knowledge/import', { method: 'POST', body: form })
  },

  saveKnowledgeEntry(payload: KnowledgeEntryInput): Promise<KnowledgeEntry> {
    return request('/api/knowledge/entries', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  },
}
