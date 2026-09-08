<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import { AlignLeft, ArrowRight, Clock3, Download, FileDown, FileText, History, Image, LoaderCircle, TextCursorInput } from 'lucide-vue-next'
import { api } from '../lib/api'
import type { TranslationResult } from '../lib/types'

const props = defineProps<{ refreshKey: number }>()
const emit = defineEmits<{ reuse: [item: TranslationResult] }>()
const items = ref<TranslationResult[]>([])
const loading = ref(true)
const error = ref('')

const kindInfo = {
  text: { label: '文本', icon: TextCursorInput },
  image: { label: '图片', icon: Image },
  document: { label: '文档', icon: FileText },
}
const languageNames = { zh: '中文', th: 'ไทย', en: 'English' }

async function load() {
  loading.value = true
  error.value = ''
  try {
    items.value = await api.listHistory()
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : '历史记录加载失败'
  } finally {
    loading.value = false
  }
}

function formatDate(value?: string) {
  if (!value) return ''
  return new Intl.DateTimeFormat('zh-CN', {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  }).format(new Date(value))
}

function shouldPreviewPdf(filename?: string | null) {
  return !!filename
    && /\.pdf$/i.test(filename)
    && /Android|iPhone|iPad|iPod|MicroMessenger/i.test(navigator.userAgent)
}

function exportUrl(item: TranslationResult) {
  return api.exportUrl(item.id, shouldPreviewPdf(item.export_filename))
}

function textExportFilename(item: TranslationResult) {
  const sourceName = item.filename || 'translation'
  return `${sourceName.replace(/\.[^.]+$/, '')}-译文.txt`
}

function canDownloadUnformatted(filename?: string | null) {
  return !!filename && /\.(pdf|docx)$/i.test(filename)
}

function unformattedExportUrl(item: TranslationResult) {
  return api.unformattedExportUrl(item.id, shouldPreviewPdf(item.filename))
}

function unformattedExportFilename(item: TranslationResult) {
  const extension = item.filename?.match(/\.[^.]+$/)?.[0] || ''
  const sourceName = item.filename || 'translation'
  return `${sourceName.replace(/\.[^.]+$/, '')}-未排版译文${extension}`
}

onMounted(load)
watch(() => props.refreshKey, load)
</script>

<template>
  <section class="page-view" aria-labelledby="history-title">
    <div class="page-heading">
      <div>
        <p class="eyebrow">RECENT ACTIVITY</p>
        <h1 id="history-title">翻译历史</h1>
        <p>最近完成的文本、图片和文档任务。</p>
      </div>
    </div>

    <p v-if="error" class="error-banner" role="alert">{{ error }}</p>
    <div v-if="loading" class="page-loading"><LoaderCircle class="spin" :size="24" /> 正在加载记录</div>
    <div v-else-if="items.length" class="history-list">
      <article v-for="item in items" :key="item.id" class="history-item">
        <div class="history-meta">
          <span class="history-kind">
            <component :is="kindInfo[item.kind || 'text'].icon" :size="15" />
            {{ kindInfo[item.kind || 'text'].label }}
          </span>
          <span>{{ languageNames[item.source_language] }} <ArrowRight :size="13" /> {{ languageNames[item.target_language] }}</span>
          <span><Clock3 :size="13" /> {{ formatDate(item.created_at) }}</span>
        </div>
        <div class="history-content">
          <p>{{ item.source_text }}</p>
          <ArrowRight class="history-arrow" :size="18" />
          <p>{{ item.translated_text }}</p>
        </div>
        <div class="history-footer">
          <span>{{ item.filename || `由 ${item.provider} 生成` }}</span>
          <div class="history-actions">
            <a
              class="text-button"
              :href="api.textExportUrl(item.id)"
              :download="textExportFilename(item)"
            >
              <Download :size="15" />TXT
            </a>
            <a
              v-if="canDownloadUnformatted(item.filename)"
              class="text-button"
              :href="unformattedExportUrl(item)"
              :download="shouldPreviewPdf(item.filename) ? undefined : unformattedExportFilename(item)"
            >
              <AlignLeft :size="15" />未排版
            </a>
            <a
              v-if="item.export_filename"
              class="text-button"
              :href="exportUrl(item)"
              :download="shouldPreviewPdf(item.export_filename) ? undefined : item.export_filename"
            >
              <FileDown :size="15" />原格式
            </a>
            <button class="text-button" type="button" @click="emit('reuse', item)">再次使用 <ArrowRight :size="15" /></button>
          </div>
        </div>
      </article>
    </div>
    <div v-else class="empty-state">
      <History :size="34" /><strong>暂无翻译记录</strong><span>完成翻译后，记录会出现在这里</span>
    </div>
  </section>
</template>
