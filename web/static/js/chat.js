/* ============================================================
   Chat — Messages, SSE streaming, file upload, token bar
   ============================================================ */
window.Chat = (function() {
var U = Utils;

// ── State ──────────────────────────────────────────────
var isStreaming = false;
var attachments = [];
var currentAbort = null;
var streamRetries = 0;
var MAX_RETRIES = 3;

function resetStreamState() {
  isStreaming = false;
  streamRetries = 0;
  if (currentAbort) { currentAbort.abort(); currentAbort = null; }
  document.getElementById('sendBtn').disabled = false;
  U.removeSendingPlaceholder();
}

// ── Message rendering ──────────────────────────────────
function renderMessages() {
  var msgs = document.getElementById('messages');
  var conv = getCurrentConv();
  msgs.innerHTML = '';
  if (!conv || !conv.messages || !conv.messages.length) {
    msgs.innerHTML = '<div id="emptyState"><div class="icon">💬</div><div class="text">新建一个对话开始聊天</div></div>';
    return;
  }
  conv.messages.forEach(function(m) { addMsgToDOM(m.role, m.content); });
  U.scrollBottom();
  updateTokenBarMsgCount(conv.messages.length);
}

function addMsgToDOM(role, content) {
  var msgs = document.getElementById('messages');
  var empty = document.getElementById('emptyState');
  if (empty) empty.remove();

  // Wrapper row for alignment
  var row = document.createElement('div');
  row.className = 'msg-row ' + role;

  // Assistant avatar + name
  if (role === 'assistant') {
    var avatarRow = document.createElement('div');
    avatarRow.className = 'msg-avatar-row';
    var agentName = '默认助手';
    if (App.currentAgentId && App.agents) {
      var ag = App.agents.find(function(a){ return a.id === App.currentAgentId; });
      if (ag) agentName = ag.name;
    }
    avatarRow.innerHTML = '<div class="msg-avatar">🤖</div><div class="msg-name">'+U.esc(agentName)+'</div>';
    row.appendChild(avatarRow);
  }

  // Message bubble
  var div = document.createElement('div');
  div.className = 'msg-bubble ' + role;

  if (content && typeof content === 'string') {
    div.innerHTML = U.renderMarkdown(content);
  } else if (content && Array.isArray(content)) {
    content.forEach(function(part) {
      if (part.type === 'text') {
        var span = document.createElement('span');
        span.innerHTML = U.renderMarkdown(part.text);
        div.appendChild(span);
      } else if (part.type === 'image_url') {
        var img = document.createElement('img');
        img.src = part.image_url.url;
        img.style.maxWidth = '300px';
        div.appendChild(img);
      }
    });
  } else {
    div.textContent = String(content || '');
  }

  row.appendChild(div);
  msgs.appendChild(row);
  U.scrollBottom();
  return div;
}

function addAssistantBubble() {
  // Wrapper row
  var row = document.createElement('div');
  row.className = 'msg-row assistant';

  // Avatar + name
  var avatarRow = document.createElement('div');
  avatarRow.className = 'msg-avatar-row';
  var agentName = '默认助手';
  if (App.currentAgentId && App.agents) {
    var ag = App.agents.find(function(a){ return a.id === App.currentAgentId; });
    if (ag) agentName = ag.name;
  }
  avatarRow.innerHTML = '<div class="msg-avatar">🤖</div><div class="msg-name">'+U.esc(agentName)+'</div>';
  row.appendChild(avatarRow);

  // Streaming bubble
  var div = document.createElement('div');
  div.className = 'msg-bubble assistant';
  div.id = 'streamingBubble';
  row.appendChild(div);
  document.getElementById('messages').appendChild(row);
  U.scrollBottom();
  return div;
}

function getCurrentConv() {
  var convs = App.conversations || [];
  for (var i = 0; i < convs.length; i++) {
    if (convs[i].id === App.currentConvId) return convs[i];
  }
  return null;
}

// ── Send message (SSE streaming) ───────────────────────
async function sendMessage() {
  var input = document.getElementById('msgInput');
  var text = input.value.trim();
  if (!text && !attachments.length) return;
  if (isStreaming) return;

  var conv = getCurrentConv();
  if (!conv) { App.newConversation(); conv = getCurrentConv(); }
  if (!conv) return;

  // Build user content
  var userContent = text;
  var imageParts = [];
  var textFiles = [];
  attachments.forEach(function(a) {
    if (a.type === 'image') imageParts.push(a.dataUrl);
    else textFiles.push('[文件: '+a.name+']\n'+a.content);
  });
  if (imageParts.length) {
    userContent = imageParts.map(function(d) { return {type:'image_url',image_url:{url:d}}; });
    if (text) userContent.push({type:'text',text:text});
  }
  if (textFiles.length) {
    userContent = textFiles.join('\n\n') + (text ? '\n\n' + text : '');
  }

  // Push user message
  conv.messages.push({role:'user', content:text});
  conv.updatedAt = Date.now();
  addMsgToDOM('user', userContent);
  input.value = '';
  attachments = [];
  renderAttachTags();

  // Disable UI
  isStreaming = true;
  document.getElementById('sendBtn').disabled = true;
  U.addSendingPlaceholder();

  // Build messages for API
  var msgs = [];
  var agentId = App.currentAgentId;
  if (agentId && App.agents) {
    var ag = App.agents.find(function(a){ return a.id === agentId; });
    if (ag && ag.systemPrompt) msgs.push({role:'system', content:ag.systemPrompt});
  }
  msgs = msgs.concat(conv.messages.slice(-40));

  // Route: Agent mode → /api/agent/stream, 对话 mode → /api/chat/stream
  var mode = App.agentMode || 'react';
  var url = mode === 'normal' ? '/api/chat/stream' : '/api/agent/stream';

  // Send
  currentAbort = new AbortController();
  try {
    await streamRequest(url, msgs, conv, mode);
  } catch (e) {
    if (e.name !== 'AbortError') {
      U.toast('连接失败: '+e.message, 'error');
      conv.messages.push({role:'assistant', content:'[网络错误，请重试]'});
      addMsgToDOM('assistant', '[网络错误，请重试]');
    }
  } finally {
    resetStreamState();
  }
  App.saveConvs();
}

async function streamRequest(url, msgs, conv, mode) {
  var body = { messages: msgs };
  if (url === '/api/agent/stream') {
    body.model = 'chat';
    body.mode = mode || 'react';
  }
  var resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type':'application/json', 'Authorization':'Bearer '+App.userToken },
    body: JSON.stringify(body),
    signal: currentAbort.signal,
  });
  if (!resp.ok) throw new Error('HTTP '+resp.status);

  var reader = resp.body.getReader();
  var decoder = new TextDecoder();
  var buffer = '';
  var bubble = addAssistantBubble();
  var fullReply = '';

  while (true) {
    var _a = await reader.read(), done = _a.done, chunk = _a.value;
    if (done) break;
    buffer += decoder.decode(chunk, {stream:true});
    var lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];
      if (!line.startsWith('data: ')) continue;
      try {
        var ev = JSON.parse(line.slice(6));
        handleSSEEvent(ev, conv, bubble, fullReply);
        if (ev.full_reply) fullReply = ev.full_reply;
        if (ev.token) fullReply += ev.token;
      } catch(e) {}
    }
  }

  if (fullReply) {
    conv.messages.push({role:'assistant', content:fullReply});
  }
  bubble.removeAttribute('id');
  updateTokenBar();
}

function handleSSEEvent(ev, conv, bubble, fullReply) {
  switch (ev.type || ev.event) {
    case 'start':
      if (ev.task_id) Agent.streamTaskId = ev.task_id;
      break;

    case 'token':
      if (fullReply !== undefined) {
        bubble.innerHTML = U.renderMarkdown(fullReply + (ev.token||''));
      }
      break;

    case 'answer':
      bubble.innerHTML = U.renderMarkdown(ev.answer || ev.content || '');
      break;

    case 'step':
      Agent.renderStep(ev);
      break;

    case 'thinking_start':
      Agent.showThinkingStream();
      break;

    case 'thinking_delta':
      Agent.appendThinking(ev.delta);
      break;

    case 'thinking_end':
      Agent.finishThinking(ev.content);
      break;

    case 'retry_attempt':
      Agent.showRetry(ev);
      break;

    case 'retry_success':
      Agent.showRetrySuccess(ev);
      break;

    case 'file_created':
      // Show prominent download card
      (function(){
        var d = document.createElement('div');
        d.className = 'step download-card';
        var token = App.userToken;
        var dlUrl = ev.download_url + (ev.download_url.indexOf('?') >= 0 ? '&' : '?') + 'token=' + encodeURIComponent(token);
        d.innerHTML = '<strong>📥 文件已生成</strong>' +
          '<div style="margin-top:6px"><a href="'+U.esc(dlUrl)+'" target="_blank" style="color:var(--accent);font-weight:600;font-size:15px">' +
          '📄 '+U.esc(ev.filename||'下载')+'</a>' +
          (ev.size_bytes ? ' <span style="font-size:12px;color:var(--slate-5)">('+ (ev.size_bytes > 1024 ? (ev.size_bytes/1024).toFixed(0)+'KB' : ev.size_bytes+'B') +')</span>' : '') +
          '</div>';
        var msgs = document.getElementById('messages');
        var bubbleEl = document.getElementById('streamingBubble');
        if (bubbleEl && bubbleEl.parentNode) { msgs.insertBefore(d, bubbleEl.parentNode); }
        else { msgs.appendChild(d); }
      })();
      break;

    case 'plan':
      if (ev.steps) {
        var html = '<div class="step plan"><strong>📋 执行计划</strong><ol style="margin:4px 0 0 16px">';
        ev.steps.forEach(function(s){ html += '<li>'+U.esc(typeof s==="string"?s:s.description||'')+'</li>'; });
        html += '</ol></div>';
        var d = document.createElement('div');
        d.innerHTML = html;
        document.getElementById('messages').appendChild(d.firstChild);
      }
      break;

    case 'confirmation_required':
      console.log('[SSE] 收到确认请求:', ev);
      Agent.showConfirmModal(ev);
      break;

    case 'intervention_applied':
      break;

    case 'feedback_digested':
      var dec = ev.decision || 'unknown';
      var d2 = document.createElement('div');
      d2.className = 'step review';
      d2.innerHTML = '<strong>📝 反馈已处理</strong> ['+U.esc(dec)+'] '+U.esc(ev.summary||'');
      document.getElementById('messages').appendChild(d2);
      break;

    case 'task_complete':
      if (ev.final_answer) {
        bubble.innerHTML = U.renderMarkdown(ev.final_answer);
      }
      break;

    case 'task_error':
    case 'task_cancelled':
      bubble.innerHTML = U.renderMarkdown('❌ '+(ev.error||'任务已取消'));
      break;

    case 'token_info':
      updateTokenBar(ev);
      break;

    case 'done':
      if (ev.full_reply) bubble.innerHTML = U.renderMarkdown(ev.full_reply);
      break;
  }
  U.scrollBottom();
}

// ── Token bar ───────────────────────────────────────────
function updateTokenBar(info, cacheInfo) {
  var bar = document.getElementById('tokenBar');
  if (!info) { bar.style.display = 'none'; return; }
  bar.style.display = 'flex';

  var pct = 0, max = 90000;
  if (info.before) pct = Math.min(100, Math.round(info.before/max*100));
  else if (info.total_tokens) pct = Math.min(100, Math.round(info.total_tokens/max*100));

  var fill = bar.querySelector('.bar-fill');
  fill.style.width = pct+'%';
  fill.className = 'bar-fill' + (pct > 80 ? ' danger' : pct > 60 ? ' warn' : '');
  document.getElementById('tokenPct').textContent = pct+'%';
  var detail = info.before ? U.estTokens(String(info.before))+' tokens' : '';
  if (info.compressed) detail += ' (已压缩)';
  document.getElementById('tokenDetail').textContent = detail;
}

function updateTokenBarMsgCount(count) {
  var bar = document.getElementById('tokenBar');
  if (count > 0) { bar.style.display = 'flex'; document.getElementById('tokenPct').textContent = count+'条'; }
}

// ── File upload ─────────────────────────────────────────
async function handleFileSelect(files) {
  if (!files || !files.length) return;
  if (attachments.length + files.length > 5) { U.toast('最多 5 个文件','warn'); return; }

  for (var i = 0; i < files.length; i++) {
    var f = files[i];
    if (f.size > 10*1024*1024) { U.toast(f.name+' 超过 10MB','warn'); continue; }

    if (f.type.startsWith('image/')) {
      var reader = new FileReader();
      reader.onload = (function(name,type) {
        return function(e) { attachments.push({name:name,type:'image',dataUrl:e.target.result}); renderAttachTags(); };
      })(f.name, f.type);
      reader.readAsDataURL(f);
    } else {
      try {
        var fd = new FormData(); fd.append('file', f);
        var resp = await fetch('/api/upload', {method:'POST',headers:{'Authorization':'Bearer '+App.userToken},body:fd});
        if (resp.ok) {
          var data = await resp.json();
          var content = data.content || data.text || '';
          attachments.push({name:f.name, type:'file', content:content});
        } else {
          U.toast(f.name+' 上传失败','error');
        }
      } catch(e) { U.toast(f.name+' 上传失败','error'); }
    }
  }
  renderAttachTags();
}

function renderAttachTags() {
  var el = document.getElementById('attachTags');
  el.innerHTML = '';
  attachments.forEach(function(a,i) {
    var tag = document.createElement('span');
    tag.className = 'attach-tag';
    tag.innerHTML = U.esc(a.name) + ' <button onclick="Chat.removeAttach('+i+')">×</button>';
    el.appendChild(tag);
  });
}

function removeAttach(i) { attachments.splice(i,1); renderAttachTags(); }

// ── Keyboard ────────────────────────────────────────────
function handleKeyDown(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}

return {
  renderMessages:renderMessages, addMsgToDOM:addMsgToDOM,
  sendMessage:sendMessage, handleKeyDown:handleKeyDown,
  handleFileSelect:handleFileSelect, removeAttach:removeAttach,
  renderAttachTags:renderAttachTags,
  updateTokenBar:updateTokenBar, updateTokenBarMsgCount:updateTokenBarMsgCount,
  get isStreaming() { return isStreaming; },
};
})();
