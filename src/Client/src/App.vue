<template>
  <div class="page-container">
    <div class="card">
      <h1 class="title">AntiDeepfake</h1>
      <p class="description">
        AntiDeepfake — сервис проверки видео на наличие подделок и дипфейков. Загрузите файл и получите результат за несколько секунд.
      </p>

      <form @submit.prevent="uploadFile">
        <input type="file" @change="handleFileChange" class="file-input" />
        <button type="submit" :disabled="!selectedFile || isLoading" class="upload-button">
          {{ isLoading ? 'Загрузка...' : 'Загрузить файл' }}
        </button>
      </form>

      <p v-if="message" class="message">{{ message }}</p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const selectedFile = ref(null)
const message = ref('')
const isLoading = ref(false)

function handleFileChange(event) {
  selectedFile.value = event.target.files[0]
}

async function uploadFile() {
  if (!selectedFile.value) return

  isLoading.value = true
  message.value = ''

  const formData = new FormData()
  formData.append('file', selectedFile.value)

  try {
    const response = await fetch('/upload', {
      method: 'POST',
      body: formData,
    })

    if (response.ok) {
      const result = await response.text()
      message.value = `✅ ${result}`
    } else {
      const error = await response.text()
      message.value = `❌ Ошибка: ${error}`
    }
  } catch (err) {
    message.value = '❌ Ошибка сети'
  } finally {
    isLoading.value = false
  }
}
</script>

<style scoped>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

* {
  font-family: 'Inter', sans-serif;
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

.page-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  background: #ffffff;
}

.card {
  background: #ffffff;
  padding: 40px;
  border-radius: 20px;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
  width: 100%;
  max-width: 420px;
  text-align: center;
  animation: fadeIn 0.8s ease-out;
}

.title {
  font-size: 32px;
  font-weight: 700;
  color: #000000;
  margin-bottom: 12px;
}

.description {
  font-size: 14px;
  color: #4a5568;
  margin-bottom: 24px;
}

.file-input {
  width: 100%;
  margin-bottom: 16px;
  padding: 10px;
  border: 2px solid #cbd5e0;
  border-radius: 8px;
  font-size: 14px;
}

.upload-button {
  width: 100%;
  padding: 12px;
  background-color: #4c51bf;
  color: #fff;
  font-weight: 600;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

.upload-button:hover:not(:disabled) {
  background-color: #434190;
}

.upload-button:disabled {
  background-color: #a0aec0;
  cursor: not-allowed;
}

.message {
  margin-top: 16px;
  font-size: 14px;
  color: #2d3748;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: scale(0.95);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}
</style>
