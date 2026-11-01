export async function request(path: string, options: RequestInit = {}) {
  const headers = new Headers(options.headers)
  const token = sessionStorage.getItem('token')
  if (token) headers.set('Authorization', 'Bearer ' + token)
  if (typeof options.body === 'string') headers.set('Content-Type','application/json')
  const res = await fetch(path, {...options, headers})
  if (!res.ok) {
    if (res.status === 401 && token) window.dispatchEvent(new Event('session-expired'))
    const detail = await res.json().catch(() => ({}))
    throw new Error(detail.message || ('请求失败 ' + res.status))
  }
  return res
}
export async function api(path:string, options?: RequestInit) {
  return (await request(path, options)).json()
}
export function post(data?: unknown): RequestInit {
  return {method:'POST', body:data === undefined ? undefined : JSON.stringify(data)}
}
