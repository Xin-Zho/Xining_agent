const BASE = ''

let token = localStorage.getItem('agent_token') || ''

export const api = {
  get token() { return token },
  set token(t: string) { token = t; localStorage.setItem('agent_token', t) },
  get username() { return localStorage.getItem('agent_user') || '' },
  set username(u: string) { localStorage.setItem('agent_user', u) },

  headers() {
    const h: Record<string, string> = { 'Content-Type': 'application/json' }
    if (token) h['Authorization'] = `Bearer ${token}`
    return h
  },

  async login(username: string, password: string) {
    const r = await fetch(`${BASE}/api/login`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    })
    const d = await r.json()
    if (!r.ok) throw new Error(d.detail || d.error || 'Login failed')
    api.token = d.token || d.access_token
    api.username = d.username || username
    return d
  },

  async register(username: string, password: string, inviteCode = '') {
    const r = await fetch(`${BASE}/api/register`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password, invite_code: inviteCode }),
    })
    const d = await r.json()
    if (!r.ok) throw new Error(d.detail || d.error || 'Register failed')
    return d
  },

  async me() {
    const r = await fetch(`${BASE}/api/auth/me`, { headers: api.headers() })
    const d = await r.json()
    if (!r.ok) throw new Error(d.detail || 'Auth failed')
    return d
  },

  async upload(file: File) {
    const fd = new FormData()
    fd.append('file', file)
    const r = await fetch(`${BASE}/api/upload`, {
      method: 'POST', headers: { Authorization: `Bearer ${token}` }, body: fd,
    })
    return r.json()
  },

  loadConvs(): any[] {
    try { return JSON.parse(localStorage.getItem(`agent_${api.username}_convs`) || '[]') }
    catch { return [] }
  },
  saveConvs(convs: any[]) {
    localStorage.setItem(`agent_${api.username}_convs`, JSON.stringify(convs.slice(0, 50)))
  },
  loadAgents(): any[] {
    try { return JSON.parse(localStorage.getItem(`agent_${api.username}_agents`) || '[]') }
    catch { return [] }
  },
  saveAgents(agents: any[]) {
    const custom = agents.filter((a: any) => !String(a.id).startsWith('_builtin_'))
    localStorage.setItem(`agent_${api.username}_agents`, JSON.stringify(custom))
  },
}
