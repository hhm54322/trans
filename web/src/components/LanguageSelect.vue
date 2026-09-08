<script setup lang="ts">
import { computed } from 'vue'
import {
  Listbox,
  ListboxButton,
  ListboxLabel,
  ListboxOption,
  ListboxOptions,
} from '@headlessui/vue'
import { Check, ChevronsUpDown } from 'lucide-vue-next'

interface LanguageOption {
  readonly value: string
  readonly label: string
}

const props = defineProps<{
  id: string
  label: string
  modelValue: string
  options: readonly LanguageOption[]
}>()

const emit = defineEmits<{
  'update:modelValue': [value: string]
}>()

const selected = computed({
  get: () => props.modelValue,
  set: (value: string) => emit('update:modelValue', value),
})

const selectedLabel = computed(() =>
  props.options.find((option) => option.value === selected.value)?.label || '',
)
</script>

<template>
  <Listbox v-model="selected" as="div" class="language-select-group">
    <ListboxLabel :for="`${id}-button`">{{ label }}</ListboxLabel>
    <div class="language-select-shell">
      <ListboxButton :id="`${id}-button`" class="language-select-trigger">
        <span>{{ selectedLabel }}</span>
        <ChevronsUpDown :size="17" aria-hidden="true" />
      </ListboxButton>
      <ListboxOptions :id="`${id}-options`" as="ul" class="language-menu">
        <ListboxOption
          v-for="option in options"
          :key="option.value"
          v-slot="{ active, selected: isSelected }"
          :value="option.value"
          as="template"
        >
          <li class="language-option" :class="{ active, selected: isSelected }">
            <Check v-if="isSelected" :size="16" aria-hidden="true" />
            <span v-else class="language-option-placeholder" aria-hidden="true" />
            <span>{{ option.label }}</span>
          </li>
        </ListboxOption>
      </ListboxOptions>
    </div>
  </Listbox>
</template>
