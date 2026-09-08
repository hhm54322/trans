<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  FileUp,
  LoaderCircle,
  Save,
  Upload,
  X,
} from 'lucide-vue-next'
import { api } from '../lib/api'
import type { KnowledgeEntry, KnowledgeImportResult } from '../lib/types'

const entries = ref<KnowledgeEntry[]>([])
const total = ref(0)
const selectedFile = ref<File | null>(null)
const loading = ref(true)
const importing = ref(false)
const saving = ref(false)
const error = ref('')
const importResult = ref<KnowledgeImportResult | null>(null)
const saveMessage = ref('')
const thaiText = ref('')
const chineseText = ref('')
const englishText = ref('')

const canImport = computed(() => !!selectedFile.value && !importing.value)
const canSave = computed(() => (
  !!thaiText.value.trim()
  && !!(chineseText.value.trim() || englishText.value.trim())
  && !saving.value
))

async function loadKnowledge() {
  loading.value = true
  error.value = ''
  try {
    const result = await api.listKnowledge()
    entries.value = result.entries
    total.value = result.total
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : '知识库加载失败'
  } finally {
    loading.value = false
  }
}

function selectFile(file?: File) {
  if (!file) return
  selectedFile.value = file
  importResult.value = null
  error.value = ''
}

async function saveEntry() {
  if (!canSave.value) return
  saving.value = true
  error.value = ''
  saveMessage.value = ''
  try {
    await api.saveKnowledgeEntry({
      thai_text: thaiText.value.trim(),
      chinese_text: chineseText.value.trim() || null,
      english_text: englishText.value.trim() || null,
    })
    thaiText.value = ''
    chineseText.value = ''
    englishText.value = ''
    saveMessage.value = '已保存到知识库'
    await loadKnowledge()
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : '知识库保存失败'
  } finally {
    saving.value = false
  }
}

async function importFile() {
  if (!selectedFile.value) return
  importing.value = true
  error.value = ''
  try {
    importResult.value = await api.importKnowledge(selectedFile.value)
    selectedFile.value = null
    await loadKnowledge()
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : '知识库导入失败'
  } finally {
    importing.value = false
  }
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat('zh-CN', {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  }).format(new Date(value))
}

onMounted(loadKnowledge)
</script>

<template>
  <section class="page-view knowledge-view" aria-labelledby="knowledge-title">
    <div class="page-heading knowledge-heading">
      <div>
        <p class="eyebrow">TERMINOLOGY</p>
        <h1 id="knowledge-title">术语知识库</h1>
      </div>
      <span class="knowledge-count">{{ total.toLocaleString() }} 条</span>
    </div>

    <section class="knowledge-manual" aria-labelledby="knowledge-manual-title">
      <div class="knowledge-section-heading">
        <h2 id="knowledge-manual-title">手动录入</h2>
        <span>泰语必填</span>
      </div>
      <div class="knowledge-entry-fields">
        <label>
          <span>ไทย <small>必填</small></span>
          <textarea v-model="thaiText" maxlength="2000" rows="3" placeholder="输入泰语内容" />
        </label>
        <label>
          <span>中文 <small>选填</small></span>
          <textarea v-model="chineseText" maxlength="5000" rows="3" placeholder="输入中文译法" />
        </label>
        <label>
          <span>English <small>选填</small></span>
          <textarea v-model="englishText" maxlength="5000" rows="3" placeholder="输入英文译法" />
        </label>
      </div>
      <div class="knowledge-manual-actions">
        <span v-if="saveMessage" class="save-message" role="status"><CheckCircle2 :size="16" />{{ saveMessage }}</span>
        <button class="primary-button" type="button" :disabled="!canSave" @click="saveEntry">
          <LoaderCircle v-if="saving" class="spin" :size="16" />
          <Save v-else :size="16" />{{ saving ? '提交中' : '提交' }}
        </button>
      </div>
    </section>

    <section class="knowledge-import" aria-labelledby="knowledge-import-title">
      <div>
        <h2 id="knowledge-import-title">导入知识库</h2>
        <p>支持参考模板格式的 Excel，也兼容现有 CSV</p>
      </div>
      <div class="knowledge-actions">
        <a class="secondary-button" :href="api.knowledgeTemplateUrl()" download="knowledge-template.xlsx">
          <Download :size="16" />模板
        </a>
        <label class="secondary-button file-picker">
          <FileUp :size="16" />选择文件
          <input class="visually-hidden" type="file" accept=".csv,.xlsx,text/csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" :disabled="importing" @change="selectFile(($event.target as HTMLInputElement).files?.[0])" />
        </label>
        <button class="primary-button" type="button" :disabled="!canImport" @click="importFile">
          <LoaderCircle v-if="importing" class="spin" :size="16" />
          <Upload v-else :size="16" />{{ importing ? '导入中' : '导入' }}
        </button>
      </div>
    </section>

    <div v-if="selectedFile" class="knowledge-file">
      <span>{{ selectedFile.name }}</span>
      <button class="icon-button" type="button" title="移除文件" aria-label="移除文件" @click="selectedFile = null"><X :size="16" /></button>
    </div>

    <p v-if="error" class="error-banner" role="alert">{{ error }}</p>
    <div v-if="importResult" class="import-summary" role="status">
      <CheckCircle2 :size="17" />
      <span>新增 {{ importResult.inserted }} 条，覆盖 {{ importResult.updated }} 条</span>
      <span v-if="importResult.invalid_rows"><AlertTriangle :size="15" />无效 {{ importResult.invalid_rows }} 条</span>
    </div>

    <div v-if="loading" class="page-loading"><LoaderCircle class="spin" :size="24" /> 正在加载知识库</div>
    <div v-else-if="entries.length" class="knowledge-table-shell">
      <table class="knowledge-table">
        <thead>
          <tr><th>ไทย</th><th>中文</th><th>English</th><th>来源</th></tr>
        </thead>
        <tbody>
          <tr v-for="entry in entries" :key="entry.id">
            <td>{{ entry.thai_text }}</td>
            <td>{{ entry.chinese_text || '-' }}</td>
            <td>{{ entry.english_text || '-' }}</td>
            <td><span class="source-file">{{ entry.source_filename }}</span><small>{{ formatDate(entry.updated_at) }}</small></td>
          </tr>
        </tbody>
      </table>
    </div>
    <div v-else class="empty-state">
      <FileUp :size="34" /><strong>知识库为空</strong>
    </div>
  </section>
</template>
