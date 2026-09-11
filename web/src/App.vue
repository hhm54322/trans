<script setup lang="ts">
import { ref } from 'vue'
import { BookOpen, History, Languages } from 'lucide-vue-next'
import HistoryView from './components/HistoryView.vue'
import KnowledgeBaseView from './components/KnowledgeBaseView.vue'
import TranslationWorkbench from './components/TranslationWorkbench.vue'
import type { TranslationResult } from './lib/types'

type ViewName = 'translate' | 'knowledge' | 'history'

const view = ref<ViewName>('translate')
const historyRefreshKey = ref(0)
const workbenchKey = ref(0)
const reusedItem = ref<TranslationResult | null>(null)

const navigation = [
  { value: 'translate' as const, label: '翻译', icon: Languages },
  { value: 'knowledge' as const, label: '知识库', icon: BookOpen },
  { value: 'history' as const, label: '历史', icon: History },
]

function selectView(next: ViewName) {
  view.value = next
}

function handleCompleted() {
  historyRefreshKey.value += 1
}

function reuse(item: TranslationResult) {
  reusedItem.value = item
  workbenchKey.value += 1
  view.value = 'translate'
}
</script>

<template>
  <div class="app-shell">
    <header class="topbar">
      <button class="brand" type="button" aria-label="返回翻译工作台" @click="selectView('translate')">
        <img class="brand-logo" src="/meta-logo.jpg" alt="1024 数字科技有限公司" />
        <span><strong>META Trans</strong><small>ZH · TH · EN</small></span>
      </button>

      <nav class="main-nav" aria-label="主导航">
        <button
          v-for="item in navigation"
          :key="item.value"
          class="nav-button"
          :class="{ active: view === item.value }"
          type="button"
          @click="selectView(item.value)"
        >
          <component :is="item.icon" :size="17" />{{ item.label }}
        </button>
      </nav>

    </header>

    <main class="app-main">
      <TranslationWorkbench
        v-show="view === 'translate'"
        :key="workbenchKey"
        :initial-result="reusedItem"
        @completed="handleCompleted"
      />
      <HistoryView
        v-if="view === 'history'"
        :refresh-key="historyRefreshKey"
        @reuse="reuse"
      />
      <KnowledgeBaseView v-else-if="view === 'knowledge'" />
    </main>

    <nav class="mobile-nav" aria-label="移动端导航">
      <button
        v-for="item in navigation"
        :key="item.value"
        :class="{ active: view === item.value }"
        type="button"
        @click="selectView(item.value)"
      >
        <component :is="item.icon" :size="20" />
        <span>{{ item.label }}</span>
      </button>
    </nav>
  </div>
</template>
