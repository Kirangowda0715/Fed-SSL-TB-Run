const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '')

export class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status }
}

async function request(path, options = {}) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), 15000)
  try {
    const response = await fetch(`${API_URL}${path}`, { ...options, signal: controller.signal })
    const body = await response.json().catch(() => null)
    if (!response.ok) throw new ApiError(body?.detail || `Request failed (${response.status})`, response.status)
    return body
  } catch (error) {
    if (error instanceof ApiError) throw error
    throw new ApiError(error.name === 'AbortError' ? 'The backend request timed out.' : 'Backend unavailable. Make sure the FastAPI server is running.')
  } finally { clearTimeout(timer) }
}

export const api = {
  metrics: () => request('/metrics'),
  status: () => request('/status'),
  metadata: () => request('/metadata'),
  predict: (file) => { const body = new FormData(); body.append('file', file); return request('/predict', { method: 'POST', body }) },
}
