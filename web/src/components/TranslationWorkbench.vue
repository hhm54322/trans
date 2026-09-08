<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue'
import {
  AlertTriangle,
  AlignLeft,
  ArrowRightLeft,
  Check,
  Copy,
  Download,
  FileDown,
  FileText,
  Image,
  Info,
  Languages,
  LoaderCircle,
  TextCursorInput,
  Trash2,
  Upload,
  X,
} from 'lucide-vue-next'
import { api } from '../lib/api'
import LanguageSelect from './LanguageSelect.vue'
import type {
  DocumentTranslationJob,
  SourceLanguage,
  TargetLanguage,
  TranslationKind,
  TranslationResult,
} from '../lib/types'

const props = defineProps<{ initialResult?: TranslationResult | null }>()
const emit = defineEmits<{ completed: [result: TranslationResult] }>()

const mode = ref<TranslationKind>(props.initialResult?.kind || 'text')
const sourceLanguage = ref<SourceLanguage>(props.initialResult?.source_language || 'auto')
const targetLanguage = ref<TargetLanguage>(props.initialResult?.target_language || 'th')
const sourceText = ref(props.initialResult?.source_text || '')
const translatedText = ref(props.initialResult?.translated_text || '')
const context = ref('')
const selectedFile = ref<File | null>(null)
const previewUrl = ref('')
const loading = ref(false)
const dragging = ref(false)
const error = ref('')
const warnings = ref<string[]>(props.initialResult?.warnings || [])
const resultMeta = ref<TranslationResult | null>(props.initialResult || null)
const copied = ref(false)
const documentJob = ref<DocumentTranslationJob | null>(null)
let activeRequestController: AbortController | null = null

const modes = [
  { value: 'text' as const, label: '文本', icon: TextCursorInput },
  { value: 'image' as const, label: '图片', icon: Image },
  { value: 'document' as const, label: '文档', icon: FileText },
]
const sourceLanguages = [
  { value: 'auto', label: '自动识别' },
  { value: 'zh', label: '简体中文' },
  { value: 'th', label: 'ไทย · 泰语' },
  { value: 'en', label: 'English · 英语' },
] as const
const targetLanguages = [
  { value: 'zh' as const, label: '简体中文' },
  { value: 'th' as const, label: 'ไทย · 泰语' },
  { value: 'en' as const, label: 'English · 英语' },
]

const acceptedFiles = computed(() =>
  mode.value === 'image' ? 'image/jpeg,image/png,image/webp' : '.docx,.pptx,.pdf,.txt,.md',
)
const fileHint = computed(() =>
  mode.value === 'image'
    ? 'JPG、PNG、WebP，最大 200 MB'
    : 'DOCX、PPTX、PDF（含扫描件）、TXT、Markdown，最大 200 MB；扫描 PDF 最多 50 页',
)
const canTranslate = computed(() =>
  !loading.value && (mode.value === 'text' ? sourceText.value.trim().length > 0 : !!selectedFile.value),
)

function changeMode(next: TranslationKind) {
  if (loading.value || mode.value === next) return
  mode.value = next
  clearWork()
}

function clearWork() {
  sourceText.value = ''
  translatedText.value = ''
  warnings.value = []
  resultMeta.value = null
  documentJob.value = null
  error.value = ''
  removeFile()
}

function swapLanguages() {
  if (sourceLanguage.value === 'auto') {
    sourceLanguage.value = targetLanguage.value
    targetLanguage.value = resultMeta.value?.source_language || 'zh'
  } else {
    const previousSource = sourceLanguage.value
    sourceLanguage.value = targetLanguage.value
    targetLanguage.value = previousSource
  }
  if (sourceText.value && translatedText.value) {
    const previousSourceText = sourceText.value
    sourceText.value = translatedText.value
    translatedText.value = previousSourceText
  }
}

function handleFile(file?: File) {
  if (!file) return
  removeFile()
  selectedFile.value = file
  if (mode.value === 'image') previewUrl.value = URL.createObjectURL(file)
  sourceText.value = ''
  translatedText.value = ''
  warnings.value = []
  resultMeta.value = null
  documentJob.value = null
  error.value = ''
}

function handleDrop(event: DragEvent) {
  dragging.value = false
  handleFile(event.dataTransfer?.files[0])
}

function removeFile() {
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = ''
  selectedFile.value = null
}

async function runTranslation() {
  if (!canTranslate.value) return
  loading.value = true
  error.value = ''
  copied.value = false
  documentJob.value = null
  try {
    const options = {
      source_language: sourceLanguage.value,
      target_language: targetLanguage.value,
      context: context.value.trim(),
    }
    const result = mode.value === 'text'
      ? await api.translateText(sourceText.value.trim(), options)
      : mode.value === 'document'
        ? await runDocumentTranslation(selectedFile.value!, options)
        : await api.translateFile(mode.value, selectedFile.value!, options)
    resultMeta.value = result
    sourceText.value = result.source_text
    translatedText.value = result.translated_text
    warnings.value = result.warnings
    emit('completed', result)
  } catch (caught) {
    if (caught instanceof Error && caught.name === 'AbortError') return
    error.value = caught instanceof Error ? caught.message : '翻译失败，请稍后重试'
  } finally {
    loading.value = false
  }
}

async function runDocumentTranslation(
  file: File,
  options: {
    source_language: SourceLanguage
    target_language: TargetLanguage
    context: string
  },
): Promise<TranslationResult> {
  const controller = new AbortController()
  activeRequestController = controller
  let pollFailures = 0

  try {
    let job = await api.createDocumentJob(file, options, controller.signal)
    documentJob.value = job

    while (job.status === 'processing') {
      await waitForNextPoll(controller.signal)
      try {
        job = await api.getDocumentJob(job.job_id, controller.signal)
        documentJob.value = job
        pollFailures = 0
      } catch (caught) {
        if (caught instanceof Error && caught.name === 'AbortError') throw caught
        pollFailures += 1
        if (pollFailures >= 3) throw caught
      }
    }

    if (job.status === 'failed') {
      throw new Error(job.error || '文档翻译失败，请稍后重试')
    }
    if (!job.result) {
      throw new Error('文档翻译已完成，但未返回译文')
    }
    return job.result
  } finally {
    if (activeRequestController === controller) activeRequestController = null
  }
}

async function waitForNextPoll(signal: AbortSignal) {
  await new Promise<void>((resolve) => window.setTimeout(resolve, 750))
  if (signal.aborted) throw new DOMException('请求已取消', 'AbortError')
}

async function copyResult() {
  if (!translatedText.value) return
  await navigator.clipboard.writeText(translatedText.value)
  copied.value = true
  window.setTimeout(() => { copied.value = false }, 1600)
}

function shouldPreviewPdf(filename: string) {
  return /\.pdf$/i.test(filename)
    && /Android|iPhone|iPad|iPod|MicroMessenger/i.test(navigator.userAgent)
}

function triggerDownload(url: string, filename: string) {
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
}

function downloadTextResult() {
  if (!translatedText.value) return
  if (resultMeta.value) {
    const sourceName = resultMeta.value.filename || selectedFile.value?.name || 'translation'
    const baseName = sourceName.replace(/\.[^.]+$/, '')
    triggerDownload(api.textExportUrl(resultMeta.value.id), `${baseName}-译文.txt`)
    return
  }
  const blob = new Blob([translatedText.value], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const baseName = selectedFile.value?.name.replace(/\.[^.]+$/, '') || 'translation'
  triggerDownload(url, `${baseName}-译文.txt`)
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}

function downloadFormatResult() {
  const result = resultMeta.value
  if (!result?.export_filename) return
  if (shouldPreviewPdf(result.export_filename)) {
    window.location.assign(api.exportUrl(result.id, true))
    return
  }
  triggerDownload(api.exportUrl(result.id), result.export_filename)
}

function canDownloadUnformatted(filename?: string | null) {
  return !!filename && /\.(pdf|docx)$/i.test(filename)
}

function downloadUnformattedResult() {
  const result = resultMeta.value
  if (!result?.filename || !canDownloadUnformatted(result.filename)) return
  const extension = result.filename.match(/\.[^.]+$/)?.[0] || ''
  const sourceName = result.filename
  const filename = `${sourceName.replace(/\.[^.]+$/, '')}-未排版译文${extension}`
  if (shouldPreviewPdf(result.filename)) {
    window.location.assign(api.unformattedExportUrl(result.id, true))
    return
  }
  triggerDownload(api.unformattedExportUrl(result.id), filename)
}

onBeforeUnmount(() => {
  activeRequestController?.abort()
  removeFile()
})
</script>

<template>
  <section aria-labelledby="workbench-title">
    <div class="workspace-heading">
      <div>
        <p class="eyebrow">ZH · TH · EN</p>
        <h1 id="workbench-title">中泰英智能翻译</h1>
      </div>
    </div>

    <div class="mode-tabs" role="tablist" aria-label="翻译类型">
      <button
        v-for="item in modes"
        :key="item.value"
        :class="{ active: mode === item.value }"
        type="button"
        :disabled="loading"
        role="tab"
        :aria-selected="mode === item.value"
        @click="changeMode(item.value)"
      >
        <component :is="item.icon" :size="17" />{{ item.label }}
      </button>
    </div>

    <div class="language-bar">
      <LanguageSelect
        id="source-language"
        v-model="sourceLanguage"
        label="源语言"
        :options="sourceLanguages"
      />
      <button class="icon-button swap-button" type="button" title="交换语言" aria-label="交换语言" @click="swapLanguages">
        <ArrowRightLeft :size="18" />
      </button>
      <LanguageSelect
        id="target-language"
        v-model="targetLanguage"
        label="目标语言"
        :options="targetLanguages"
      />
    </div>

    <label class="context-field">
      <span><Info :size="15" /> 本次翻译背景 <small>选填</small></span>
      <textarea
        v-model="context"
        maxlength="5000"
        rows="2"
        placeholder="例如：酒店预订合同；META Trans 是产品名，保留英文"
      />
    </label>

    <div class="translation-grid">
      <section class="editor-panel source-panel" aria-label="原文">
        <header class="panel-heading">
          <span>{{ mode === 'text' ? '原文' : '待翻译文件' }}</span>
          <button v-if="sourceText || selectedFile" class="icon-button" type="button" title="清空" aria-label="清空" :disabled="loading" @click="clearWork">
            <Trash2 :size="17" />
          </button>
        </header>

        <textarea
          v-if="mode === 'text'"
          v-model="sourceText"
          maxlength="50000"
          placeholder="输入要翻译的内容"
          aria-label="要翻译的原文"
        />

        <template v-else>
          <label
            v-if="!selectedFile"
            class="drop-zone"
            :class="{ active: dragging }"
            @dragenter.prevent="dragging = true"
            @dragover.prevent="dragging = true"
            @dragleave.prevent="dragging = false"
            @drop.prevent="handleDrop"
          >
            <input class="visually-hidden" type="file" :accept="acceptedFiles" :disabled="loading" @change="handleFile(($event.target as HTMLInputElement).files?.[0])" />
            <span class="upload-icon"><Upload :size="23" /></span>
            <strong>选择或拖入{{ mode === 'image' ? '图片' : '文档' }}</strong>
            <span>{{ fileHint }}</span>
          </label>
          <div v-else class="selected-file">
            <img v-if="previewUrl" :src="previewUrl" alt="待翻译图片预览" />
            <span v-else class="file-icon"><FileText :size="25" /></span>
            <div>
              <strong>{{ selectedFile.name }}</strong>
              <span>{{ (selectedFile.size / 1024 / 1024).toFixed(2) }} MB</span>
            </div>
            <button class="icon-button" type="button" title="移除文件" aria-label="移除文件" :disabled="loading" @click="removeFile"><X :size="18" /></button>
          </div>
          <div v-if="sourceText" class="extracted-source">
            <span>识别原文</span>
            <p>{{ sourceText }}</p>
          </div>
        </template>

        <footer class="panel-footer">
          <span v-if="mode === 'text'">{{ sourceText.length.toLocaleString() }} / 50,000</span>
          <span v-else>{{ selectedFile ? selectedFile.name : fileHint }}</span>
        </footer>
      </section>

      <section class="editor-panel result-panel" aria-label="译文">
        <header class="panel-heading">
          <span>译文</span>
          <div class="panel-tools">
            <button class="icon-button" type="button" :title="copied ? '已复制' : '复制译文'" :aria-label="copied ? '已复制' : '复制译文'" :disabled="!translatedText" @click="copyResult">
              <Check v-if="copied" :size="17" /><Copy v-else :size="17" />
            </button>
            <button class="download-tool" type="button" title="下载纯文本译文" :disabled="!translatedText" @click="downloadTextResult">
              <Download :size="15" />TXT
            </button>
            <button v-if="canDownloadUnformatted(resultMeta?.filename)" class="download-tool" type="button" title="下载未排版译文" :disabled="!translatedText" @click="downloadUnformattedResult">
              <AlignLeft :size="15" />未排版
            </button>
            <button v-if="resultMeta?.export_filename" class="download-tool" type="button" :title="`下载 ${resultMeta.export_filename}`" :disabled="!translatedText" @click="downloadFormatResult">
              <FileDown :size="15" />原格式
            </button>
          </div>
        </header>

        <div v-if="loading" class="result-empty loading-state" aria-live="polite">
          <LoaderCircle class="spin" :size="28" />
          <strong>{{ mode === 'document' ? (documentJob?.message || '正在上传并解析文档') : '正在翻译' }}</strong>
          <div
            v-if="mode === 'document' && documentJob"
            class="translation-progress"
            role="progressbar"
            :aria-valuenow="documentJob.progress"
            aria-valuemin="0"
            aria-valuemax="100"
          >
            <div class="translation-progress-meta">
              <span>已完成 {{ documentJob.completed_pages }} / {{ documentJob.total_pages }} 页</span>
              <span>{{ documentJob.progress }}%</span>
            </div>
            <div class="translation-progress-track">
              <span :style="{ width: `${documentJob.progress}%` }" />
            </div>
          </div>
        </div>
        <textarea
          v-else-if="translatedText"
          v-model="translatedText"
          aria-label="翻译结果"
        />
        <div v-else class="result-empty">
          <Languages :size="29" />
          <strong>译文将在这里显示</strong>
        </div>

        <footer class="panel-footer result-footer">
          <span>{{ translatedText.length.toLocaleString() }} 字符</span>
          <span v-if="resultMeta">{{ resultMeta.provider }}</span>
        </footer>
      </section>
    </div>

    <p v-if="error" class="error-banner" role="alert">{{ error }}</p>

    <div class="action-row">
      <button class="secondary-button" type="button" :disabled="loading || (!sourceText && !selectedFile)" @click="clearWork">
        <Trash2 :size="17" />清空
      </button>
      <button class="primary-button" type="button" :disabled="!canTranslate" @click="runTranslation">
        <LoaderCircle v-if="loading" class="spin" :size="17" />
        <Languages v-else :size="17" />
        {{ loading ? '翻译中' : '开始翻译' }}
      </button>
    </div>

    <section v-if="warnings.length" class="warning-section" aria-label="质量提示">
      <div class="insight-title"><AlertTriangle :size="16" />质量提示</div>
      <p v-for="warning in warnings" :key="warning">{{ warning }}</p>
    </section>
  </section>
</template>
