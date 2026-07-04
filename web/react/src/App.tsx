import { useState, useEffect, useRef } from 'react'
import { Message, MessageContent, MessageResponse } from '@/components/ai-elements/message'
import { Conversation, ConversationContent, ConversationEmptyState } from '@/components/ai-elements/conversation'
import { PromptInput, PromptInputTextarea, PromptInputSubmit, PromptInputBody, PromptInputFooter, PromptInputTools, PromptInputButton } from '@/components/ai-elements/prompt-input'
import { Reasoning, ReasoningTrigger, ReasoningContent } from '@/components/ai-elements/reasoning'
import { ChainOfThought, ChainOfThoughtHeader, ChainOfThoughtStep, ChainOfThoughtContent } from '@/components/ai-elements/chain-of-thought'
import { Confirmation } from '@/components/ai-elements/confirmation'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { TooltipProvider } from '@/components/ui/tooltip'
import { MessageSquare, Plus, Bot, BarChart3, LogOut, Paperclip, Brain } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { api } from './api'
import type { AgentStep, ToolExecution } from './types'

const genId = () => 'x' + Date.now() + Math.random().toString(36).slice(2, 8)

export default function App() {
  const [loggedIn, setLoggedIn] = useState(false)
  const [authUser, setAuthUser] = useState('')
  const [authPass, setAuthPass] = useState('')
  const [authError, setAuthError] = useState('')
  const [isRegister, setIsRegister] = useState(false)
  const [authCode, setAuthCode] = useState('')

  const [convs, setConvs] = useState<any[]>([])
  const [currentConvId, setCurrentConvId] = useState<string | null>(null)
  const [agents, setAgents] = useState<any[]>([])
  const [currentAgentId, setCurrentAgentId] = useState<string | null>(null)
  const [agentMode, setAgentMode] = useState<string>('react')
  const [fontSize, setFontSize] = useState<string>('normal')
  const [showEval, setShowEval] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const [streaming, setStreaming] = useState(false)
  const [streamingContent, setStreamingContent] = useState('')
  const [agentSteps, setAgentSteps] = useState<AgentStep[]>([])
  const [thinkingText, setThinkingText] = useState('')
  const [showThinking, setShowThinking] = useState(false)
  const [confirmation, setConfirmation] = useState<ToolExecution | null>(null)
  const [confirmationTimer, setConfirmationTimer] = useState(60)

  const abortRef = useRef<AbortController | null>(null)

  // Init
  useEffect(() => {
    const t = localStorage.getItem('agent_token')
    const u = localStorage.getItem('agent_user')
    if (t && u) {
      api.token = t; api.username = u
      fetch('/api/auth/me', { headers: { 'Authorization': `Bearer ${t}` } })
        .then(r => r.json())
        .then(d => { if (d.username) { setAuthUser(d.username); setLoggedIn(true); initData() } })
        .catch(() => { api.token = ''; api.username = '' })
    }
    const f = localStorage.getItem('agent_font') || 'normal'
    setFontSize(f)
    document.body.classList.toggle('text-large', f === 'large')
  }, [])

  function initData() {
    const c = api.loadConvs()
    setConvs(c)
    if (c.length) setCurrentConvId(c[c.length - 1].id)
    const saved = api.loadAgents()
    setAgents(saved)
  }

  function getCurrentConv() { return convs.find(c => c.id === currentConvId) || null }

  function displayMsgs() {
    const conv = getCurrentConv()
    return conv?.messages?.map((m: any, i: number) => ({ ...m, key: i })) || []
  }

  // Auth
  async function handleAuth() {
    setAuthError('')
    try {
      if (isRegister) await api.register(authUser, authPass, authCode)
      await api.login(authUser, authPass)
      setLoggedIn(true)
      initData()
    } catch (e: any) { setAuthError(e.message) }
  }

  function logout() {
    api.token = ''; api.username = ''
    localStorage.removeItem('agent_token'); localStorage.removeItem('agent_user')
    setLoggedIn(false)
  }

  // Conv
  function newConv() {
    const conv = { id: genId(), title: '新对话', messages: [], updatedAt: Date.now() }
    const updated = [...convs, conv]
    setConvs(updated); setCurrentConvId(conv.id)
    api.saveConvs(updated)
    setAgentSteps([]); setStreamingContent('')
  }

  function deleteConv(id: string) {
    const updated = convs.filter(c => c.id !== id)
    setConvs(updated)
    if (currentConvId === id) setCurrentConvId(updated.length ? updated[updated.length - 1].id : null)
    api.saveConvs(updated)
  }

  // Send with SSE
  async function doSend(text: string) {
    if (streaming || !text.trim()) return
    let conv = getCurrentConv()
    if (!conv) { newConv(); const c = getCurrentConv(); if (!c) return; conv = c }

    conv.messages.push({ role: 'user', content: text })
    conv.updatedAt = Date.now()
    api.saveConvs(convs)

    setStreaming(true); setStreamingContent(''); setAgentSteps([])
    setThinkingText(''); setShowThinking(false)

    const msgs: any[] = []
    if (currentAgentId) {
      const ag = agents.find(a => a.id === currentAgentId)
      if (ag?.systemPrompt) msgs.push({ role: 'system', content: ag.systemPrompt })
    }
    msgs.push(...conv.messages.slice(-40))

    const url = agentMode === 'normal' ? '/api/chat/stream' : '/api/agent/stream'
    const body: any = { messages: msgs }
    if (url === '/api/agent/stream') { body.model = 'chat'; body.mode = agentMode || 'react' }

    abortRef.current = new AbortController()
    try {
      const resp = await fetch(url, {
        method: 'POST', headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${api.token}` },
        body: JSON.stringify(body), signal: abortRef.current.signal,
      })
      const reader = resp.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = '', fullReply = ''
      const steps: AgentStep[] = []

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const ev = JSON.parse(line.slice(6))
            const t = ev.type || ev.event
            if (t === 'token') { fullReply += ev.token || ''; setStreamingContent(fullReply) }
            else if (t === 'answer') { fullReply = ev.answer || ev.content || ''; setStreamingContent(fullReply) }
            else if (t === 'step') {
              const s = ev.step || ev
              steps.push({ type: s.type || 'tool_call', tool_name: s.tool_name, observation: s.observation, tool_args: s.tool_args, thought: s.thought, duration_ms: s.duration_ms, status: s.status })
              setAgentSteps([...steps])
            }
            else if (t === 'thinking_start') { setShowThinking(true); setThinkingText('') }
            else if (t === 'thinking_delta') { setThinkingText(p => p + (ev.delta || '')) }
            else if (t === 'thinking_end') { setShowThinking(false) }
            else if (t === 'file_created') { fullReply += `\n\n📥 [${ev.filename}](${ev.download_url})`; setStreamingContent(fullReply) }
            else if (t === 'confirmation_required') {
              setConfirmation({ task_id: ev.task_id, step_num: ev.step_num, tool_name: ev.tool_name, args: typeof ev.args === 'string' ? JSON.parse(ev.args) : (ev.args || {}) })
              setConfirmationTimer(60)
            }
            else if (t === 'done') { if (ev.full_reply) { fullReply = ev.full_reply; setStreamingContent(fullReply) } }
            else if (t === 'task_error' || t === 'task_cancelled') { setStreamingContent('❌ ' + (ev.error || 'Cancelled')) }
          } catch { }
        }
      }
      if (fullReply) {
        conv.messages.push({ role: 'assistant', content: fullReply, steps })
        api.saveConvs(convs)
        setConvs([...convs])
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') setStreamingContent('[连接失败，请重试]')
    } finally {
      setStreaming(false); abortRef.current = null
    }
  }

  async function confirmTool(approved: boolean) {
    if (!confirmation) return
    try {
      await fetch(`/api/agent/confirm/${confirmation.task_id}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ step_number: confirmation.step_num, approved }),
      })
    } catch { }
    setConfirmation(null)
  }

  useEffect(() => {
    if (!confirmation) return
    const id = setInterval(() => setConfirmationTimer(p => {
      if (p <= 1) { confirmTool(false); return 60 }
      return p - 1
    }), 1000)
    return () => clearInterval(id)
  }, [confirmation])

  function onFontChange(v: string) {
    setFontSize(v); localStorage.setItem('agent_font', v)
    document.body.classList.toggle('text-large', v === 'large')
  }

  function esc(s: string) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;') }

  function renderMd(text: string) {
    if (!text) return ''
    let h = esc(text)
    h = h.replace(/```(\w*)\n([\s\S]*?)```/g, (_, _l, c) => `<pre><code>${c.trimEnd()}</code></pre>`)
    h = h.replace(/`([^`]+)`/g, '<code>$1</code>')
    h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    h = h.replace(/\n/g, '<br>')
    return h
  }

  const messages = displayMsgs()

  // Auth page
  if (!loggedIn) {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <div className="w-full max-w-sm space-y-4 rounded-xl border bg-card p-8 shadow-sm">
          <h1 className="text-2xl font-bold text-center">{isRegister ? '注册' : '登录'}</h1>
          <p className="text-sm text-muted-foreground text-center">{isRegister ? '创建新账号' : '欢迎回来'}</p>
          <input className="flex h-10 w-full rounded-md border bg-background px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-ring" placeholder="用户名" value={authUser} onChange={e => setAuthUser(e.target.value)} />
          <input className="flex h-10 w-full rounded-md border bg-background px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-ring" type="password" placeholder="密码" value={authPass} onChange={e => setAuthPass(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleAuth()} />
          {isRegister && <input className="flex h-10 w-full rounded-md border bg-background px-3 py-2 text-sm outline-none" placeholder="邀请码" value={authCode} onChange={e => setAuthCode(e.target.value)} />}
          {authError && <p className="text-sm text-destructive text-center">{authError}</p>}
          <Button className="w-full" onClick={handleAuth}>{isRegister ? '注 册' : '登 录'}</Button>
          <p className="text-sm text-center text-muted-foreground">{isRegister ? '已有账号？' : '没有账号？'}
            <button className="ml-1 text-primary font-medium" onClick={() => { setIsRegister(!isRegister); setAuthError('') }}>{isRegister ? '登录' : '注册'}</button>
          </p>
        </div>
      </div>
    )
  }

  // Main app
  return (
    <TooltipProvider>
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <aside className={`${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} md:translate-x-0 fixed md:relative z-50 flex h-full w-[250px] min-w-[250px] flex-col border-r bg-sidebar transition-transform`}>
        <div className="p-4 border-b">
          <Button className="w-full" onClick={newConv}><Plus className="mr-2 size-4" /> 新建对话</Button>
        </div>
        <div className="flex-1 overflow-y-auto p-2">
          <div className="px-2 py-2 text-[11px] font-semibold uppercase text-muted-foreground tracking-wider">对话</div>
          {[...convs].sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0)).map(c => (
            <div key={c.id} className={`flex items-center gap-2 rounded-lg px-2 py-2 text-sm cursor-pointer transition-colors group ${c.id === currentConvId ? 'bg-primary text-primary-foreground' : 'hover:bg-accent'}`}
              onClick={() => { setCurrentConvId(c.id); setAgentSteps([]); setStreamingContent('') }}>
              <MessageSquare className="size-4 shrink-0" />
              <span className="flex-1 truncate">{c.title || '新对话'}</span>
              <button className="hidden group-hover:flex size-5 items-center justify-center rounded-full text-xs opacity-60 hover:opacity-100"
                onClick={e => { e.stopPropagation(); deleteConv(c.id) }}>×</button>
            </div>
          ))}
          <div className="px-2 py-2 text-[11px] font-semibold uppercase text-muted-foreground tracking-wider mt-4">AGENT</div>
          <div className={`flex items-center gap-2 rounded-lg px-2 py-2 text-sm cursor-pointer transition-colors ${!currentAgentId ? 'bg-primary text-primary-foreground' : 'hover:bg-accent'}`}
            onClick={() => setCurrentAgentId(null)}>
            <Bot className="size-4 shrink-0" /> <span className="flex-1 truncate">默认助手</span>
          </div>
          {agents.map(a => (
            <div key={a.id} className={`flex items-center gap-2 rounded-lg px-2 py-2 text-sm cursor-pointer transition-colors ${a.id === currentAgentId ? 'bg-primary text-primary-foreground' : 'hover:bg-accent'}`}
              onClick={() => setCurrentAgentId(a.id)}>
              <Bot className="size-4 shrink-0" /> <span className="flex-1 truncate">{a.name}</span>
            </div>
          ))}
        </div>
        <div className="flex items-center gap-2 border-t p-3 text-xs text-muted-foreground">
          <span className="flex-1 truncate font-medium text-foreground">{api.username}</span>
          <Select value={fontSize} onValueChange={onFontChange}>
            <SelectTrigger className="h-7 w-12 border-none bg-transparent text-xs shadow-none"><SelectValue /></SelectTrigger>
            <SelectContent><SelectItem value="normal">A</SelectItem><SelectItem value="large">A+</SelectItem></SelectContent>
          </Select>
          <button className="text-xs hover:text-foreground" onClick={() => setShowEval(!showEval)} title="评估"><BarChart3 className="size-3" /></button>
          <button className="text-xs hover:text-destructive" onClick={logout}><LogOut className="size-3" /></button>
        </div>
      </aside>

      {sidebarOpen && <div className="fixed inset-0 z-40 bg-black/30 md:hidden" onClick={() => setSidebarOpen(false)} />}

      {/* Main */}
      <main className="flex flex-1 flex-col overflow-hidden">
        <header className="flex h-14 items-center gap-3 border-b bg-card px-4 shrink-0">
          <button className="md:hidden" onClick={() => setSidebarOpen(true)}>☰</button>
          <span className="font-bold text-lg">Xin</span>
          {currentAgentId && <Badge variant="secondary">{agents.find(a => a.id === currentAgentId)?.name || 'Agent'}</Badge>}
          <div className="flex-1" />
          <Select value={agentMode} onValueChange={setAgentMode}>
            <SelectTrigger className="h-8 w-32 rounded-full border text-xs"><SelectValue /></SelectTrigger>
            <SelectContent><SelectItem value="react">🤖 Agent</SelectItem><SelectItem value="normal">💬 对话</SelectItem></SelectContent>
          </Select>
        </header>

        {!showEval ? (
          <Conversation className="flex-1">
            <ConversationContent>
              {messages.length === 0 && !streaming && (
                <ConversationEmptyState title="新建一个对话开始聊天" description="选 Agent 模式可使用搜索、命令等工具" />
              )}
              {/* Past messages */}
              {messages.map((m: any, i: number) => (
                <Message key={m.key || i} from={m.role}>
                  {m.role === 'assistant' && m.steps?.length > 0 && (
                    <ChainOfThought className="mb-3">
                      <ChainOfThoughtHeader>
                        <Brain className="size-4" />
                        <span>工具 {m.steps.filter((s: any) => s.type === 'tool_call').length} · 思考 {m.steps.filter((s: any) => s.type === 'thought').length}</span>
                      </ChainOfThoughtHeader>
                      <ChainOfThoughtContent>
                        {m.steps.slice(0, 5).map((s: any, si: number) => (
                          <ChainOfThoughtStep key={si}>
                            <span className="text-xs text-muted-foreground">
                              {s.type === 'tool_call' ? `🔧 ${s.tool_name}${s.duration_ms ? ` (${s.duration_ms}ms)` : ''}` : `💡 ${s.thought?.slice(0, 80) || ''}`}
                            </span>
                          </ChainOfThoughtStep>
                        ))}
                      </ChainOfThoughtContent>
                    </ChainOfThought>
                  )}
                  <MessageContent>
                    <MessageResponse><ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>{m.content || ''}</ReactMarkdown></MessageResponse>
                  </MessageContent>
                </Message>
              ))}
              {/* Streaming message */}
              {streaming && (
                <Message from="assistant">
                  {showThinking && (
                    <Reasoning isStreaming={showThinking}>
                      <ReasoningTrigger />
                      <ReasoningContent>{thinkingText}</ReasoningContent>
                    </Reasoning>
                  )}
                  {agentSteps.length > 0 && (
                    <ChainOfThought className="mb-3">
                      <ChainOfThoughtHeader><Brain className="size-4" /><span>工具调用中 ({agentSteps.length})</span></ChainOfThoughtHeader>
                      <ChainOfThoughtContent>
                        {agentSteps.map((s, si) => (
                          <ChainOfThoughtStep key={si}>
                            <span className="text-xs text-muted-foreground">
                              {s.type === 'tool_call' ? `🔧 ${s.tool_name}${s.status === 'failed' ? ' ❌' : ''}${s.duration_ms ? ` (${s.duration_ms}ms)` : ''}` : `💡 ${s.thought?.slice(0, 80) || ''}`}
                            </span>
                          </ChainOfThoughtStep>
                        ))}
                      </ChainOfThoughtContent>
                    </ChainOfThought>
                  )}
                  <MessageContent>
                    <MessageResponse>{streamingContent}</MessageResponse>
                  </MessageContent>
                </Message>
              )}
            </ConversationContent>
          </Conversation>
        ) : (
          <div className="flex-1 overflow-y-auto p-6">
            <div className="flex items-center gap-3 mb-6">
              <Button variant="ghost" size="icon" onClick={() => setShowEval(false)}>←</Button>
              <h2 className="text-lg font-bold">Agent 评估</h2>
            </div>
            <p className="text-muted-foreground">评估仪表盘通过后端 API 加载，敬请期待...</p>
          </div>
        )}

        {/* Input */}
        <div className="border-t bg-card p-4 shrink-0">
          <PromptInput onSubmit={({ text }) => doSend(text)}>
            <PromptInputBody>
              <PromptInputTextarea placeholder="输入消息...（Shift+Enter 换行）" className="min-h-10" />
            </PromptInputBody>
            <PromptInputFooter>
              <PromptInputTools>
                <PromptInputButton tooltip="上传文件" onClick={() => document.getElementById('fu')?.click()}>
                  <Paperclip className="size-4" />
                </PromptInputButton>
              </PromptInputTools>
              <PromptInputSubmit status={streaming ? 'streaming' : 'ready'} />
            </PromptInputFooter>
          </PromptInput>
          <input type="file" id="fu" className="hidden" multiple onChange={async (e) => {
            const files = e.currentTarget.files; if (!files) return
            for (const f of Array.from(files)) {
              try {
                if (f.type.startsWith('image/')) {
                  const reader = new FileReader()
                  reader.onload = () => {}
                  reader.readAsDataURL(f)
                } else { await api.upload(f) }
              } catch { }
            }
            e.currentTarget.value = ''
          }} />
        </div>
      </main>

      {/* Confirmation modal */}
      <Confirmation
        open={!!confirmation}
        onOpenChange={(open) => { if (!open) confirmTool(false) }}
        title={confirmation ? `确认执行: ${confirmation.tool_name}` : ''}
        description={confirmation ? JSON.stringify(confirmation.args, null, 2) : ''}
        timer={confirmation ? `${confirmationTimer}s 后自动拒绝` : ''}
        onConfirm={() => confirmTool(true)}
        onCancel={() => confirmTool(false)}
      />
    </div>
    </TooltipProvider>
  )
}
