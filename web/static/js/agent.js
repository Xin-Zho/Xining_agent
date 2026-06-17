/* ============================================================
   Agent — Step rendering, thinking stream, retry, confirm
   ============================================================ */
window.Agent = (function() {
var U = Utils;
var streamTaskId = null;
var confirmTaskId = null;
var confirmStep = null;
var confirmTimer = null;
var thinkingStreamEl = null;
var thinkingBuffer = '';
var currentRound = 0;

// ── Thinking stream ────────────────────────────────────
function showThinkingStream() {
  var msgs = document.getElementById('messages');
  currentRound++;
  // Round separator
  var round = document.createElement('div');
  round.className = 'step round-header';
  round.textContent = '第 ' + currentRound + ' 轮';
  msgs.appendChild(round);

  thinkingStreamEl = document.createElement('div');
  thinkingStreamEl.className = 'thinking-stream';
  thinkingStreamEl.innerHTML = '<div class="label">💭 Agent 思考中 <span class="dots">...</span></div><div class="thinking-text"></div>';
  msgs.appendChild(thinkingStreamEl);
  thinkingBuffer = '';
  U.scrollBottom();
}

function appendThinking(delta) {
  if (!thinkingStreamEl) return;
  thinkingBuffer += delta;
  var textEl = thinkingStreamEl.querySelector('.thinking-text');
  if (textEl) textEl.textContent = thinkingBuffer.slice(-500);
  U.scrollBottom();
}

function finishThinking(content) {
  // If active streaming UI exists, collapse it
  if (thinkingStreamEl) {
    thinkingStreamEl.classList.add('collapsed');
    var summary = (content || thinkingBuffer || '').slice(0, 200);
    thinkingStreamEl.innerHTML = '<div class="label">💭 已思考</div><div>'+U.esc(summary)+(summary.length>=200?'...':'')+'</div>';
    thinkingStreamEl.onclick = function() { this.classList.toggle('collapsed'); };
    thinkingStreamEl = null;
  } else {
    // No active stream — render thinking card directly
    var msgs = document.getElementById('messages');
    var bubble = document.getElementById('streamingBubble');
    var div = document.createElement('div');
    div.className = 'step thinking-card';
    var text = (content || thinkingBuffer || '').slice(0, 500);
    div.innerHTML = '<strong>💭 思考</strong><div style="margin-top:4px;font-size:13px;color:var(--slate-6)">'+U.esc(text)+(text.length>=500?'...':'')+'</div>';
    // Insert before streamingBubble so answer stays at bottom
    if (bubble) {
      msgs.insertBefore(div, bubble);
    } else {
      msgs.appendChild(div);
    }
  }
  thinkingBuffer = '';
  U.scrollBottom();
}

// ── Retry indicators ───────────────────────────────────
function showRetry(ev) {
  var msgs = document.getElementById('messages');
  var div = document.createElement('div');
  div.className = 'retry-indicator';
  div.id = 'retry-'+ev.tool_name;
  div.innerHTML = '🔄 ' + U.esc(ev.tool_name) + ' 重试中 ('+ev.attempt+'/'+ev.max_retries+')... 等待 '+ev.wait_seconds+'s';
  msgs.appendChild(div);
  U.scrollBottom();
}

function showRetrySuccess(ev) {
  var el = document.getElementById('retry-'+ev.tool_name);
  if (el) {
    el.className = 'retry-indicator success';
    el.innerHTML = '✅ ' + U.esc(ev.tool_name) + ' 重试成功 (第'+(ev.attempt||'')+'次)';
    setTimeout(function(){ if(el.parentNode) el.remove(); }, 5000);
  }
}

// ── Step rendering ─────────────────────────────────────
function renderStep(ev) {
  var msgs = document.getElementById('messages');
  var empty = document.getElementById('emptyState');
  if (empty) empty.remove();

  var stype = ev.step_type || ev.type || '';
  var div = document.createElement('div');
  div.className = 'step ' + stype;

  // Insert before streamingBubble so final answer stays at bottom
  var bubble = document.getElementById('streamingBubble');

  if (stype === 'tool_call') {
    var tool = ev.tool_name || '';
    var tagClass = U.toolTagClass(tool);
    var time = U.formatDuration(ev.duration_ms);
    var argsStr = typeof ev.args === 'string' ? ev.args : JSON.stringify(ev.args||{}, null, 2);
    var resultStr = typeof ev.result === 'string' ? ev.result : JSON.stringify(ev.result||ev.observation||{}, null, 2);

    div.innerHTML =
      '<div class="tool-header">'+
        '<span class="tool-tag '+tagClass+'">🔧 '+U.esc(tool)+'</span>'+
        (time ? '<span class="tool-time">⏱ '+time+'</span>' : '')+
      '</div>'+
      '<div class="tool-args" onclick="var c=this.querySelector(\'.tool-args-content\');c.classList.toggle(\'show\');var t=this.querySelector(\'.toggle\');t.textContent=c.classList.contains(\'show\')?\'▲ 收起\':\'▶ 展开\'">'+
        '参数 <span class="toggle">▶ 展开</span>'+
        '<div class="tool-args-content"><pre style="margin:4px 0;font-size:11px">'+U.esc(argsStr.slice(0,500))+'</pre></div>'+
      '</div>'+
      (resultStr && resultStr !== '{}' ?
        '<div class="tool-result">'+U.renderMarkdown(resultStr.slice(0,500))+'</div>' : '');

    if (ev.status === 'failed' || (ev.result && typeof ev.result === 'string' && ev.result.includes('error'))) {
      div.style.borderColor = 'var(--danger)';
      div.style.background = '#FEF2F2';
    }
  } else if (stype === 'thought') {
    var thought = ev.thought || ev.message || ev.content || '';
    div.innerHTML = '<strong>💡 思考</strong><div style="margin-top:4px">'+U.renderMarkdown(thought.slice(0,500))+'</div>';
  } else if (stype === 'plan') {
    div.innerHTML = '<strong>📋 规划</strong><div style="margin-top:4px">'+U.renderMarkdown((ev.plan||'').slice(0,500))+'</div>';
  } else if (stype === 'review') {
    div.className = 'step review';
    var parts = [];
    if (ev.missing_steps && ev.missing_steps.length) parts.push('<span style="color:var(--danger)">⚠ 缺失步骤: '+ev.missing_steps.length+'</span>');
    if (ev.flawed_logic && ev.flawed_logic.length) parts.push('<span style="color:var(--warning)">⚠ 逻辑缺陷: '+ev.flawed_logic.length+'</span>');
    if (ev.boundary_gaps && ev.boundary_gaps.length) parts.push('<span style="color:var(--info)">⚠ 边界缺口: '+ev.boundary_gaps.length+'</span>');
    if (ev.suggestions && ev.suggestions.length) parts.push('<span>💡 建议: '+ev.suggestions.length+' 条</span>');
    div.innerHTML = '<strong>🔍 审查结果</strong><div style="margin-top:4px">'+parts.join(' &middot; ')+'</div>';
  } else {
    div.innerHTML = '<strong>'+U.esc(stype)+'</strong>';
  }

  // Insert before streamingBubble so answer stays at bottom
  if (bubble) {
    msgs.insertBefore(div, bubble);
  } else {
    msgs.appendChild(div);
  }
  U.scrollBottom();
}

// ── Confirm modal ───────────────────────────────────────
function showConfirmModal(ev) {
  confirmTaskId = ev.task_id || streamTaskId;
  confirmStep = ev.step_num;
  document.getElementById('confirmToolName').textContent = '工具: ' + (ev.tool_name || '');
  document.getElementById('confirmArgs').textContent = JSON.stringify(ev.args||{}, null, 2);
  document.getElementById('confirmOverlay').classList.add('show');

  var timerEl = document.getElementById('confirmTimer');
  var secs = 60;
  timerEl.textContent = secs + 's 后自动拒绝';
  if (confirmTimer) clearInterval(confirmTimer);
  confirmTimer = setInterval(function() {
    secs--;
    timerEl.textContent = secs + 's 后自动拒绝';
    if (secs <= 0) { clearInterval(confirmTimer); confirmTool(false); }
  }, 1000);
}

function confirmTool(approved) {
  if (confirmTimer) clearInterval(confirmTimer);
  document.getElementById('confirmOverlay').classList.remove('show');
  if (confirmTaskId && confirmStep) {
    fetch('/api/agent/confirm/'+confirmTaskId, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({step_number:confirmStep, approved:approved}),
    }).catch(function(){});
  }
  confirmTaskId = null;
  confirmStep = null;
}

return {
  get streamTaskId() { return streamTaskId; }, set streamTaskId(v) { streamTaskId = v; },
  showThinkingStream:showThinkingStream,
  appendThinking:appendThinking,
  finishThinking:finishThinking,
  showRetry:showRetry,
  showRetrySuccess:showRetrySuccess,
  renderStep:renderStep,
  showConfirmModal:showConfirmModal,
  confirmTool:confirmTool,
};
})();
