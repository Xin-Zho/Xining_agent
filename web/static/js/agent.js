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
  var bubble = document.getElementById('streamingBubble');
  if (bubble) { msgs.insertBefore(round, bubble && bubble.parentNode ? bubble.parentNode : bubble); }
  else { msgs.appendChild(round); }

  thinkingStreamEl = document.createElement('div');
  thinkingStreamEl.className = 'thinking-stream';
  thinkingStreamEl.innerHTML = '<div class="label">💭 Agent 思考中 <span class="dots">...</span></div><div class="thinking-text"></div>';
  if (bubble) { msgs.insertBefore(thinkingStreamEl, bubble && bubble.parentNode ? bubble.parentNode : bubble); }
  else { msgs.appendChild(thinkingStreamEl); }
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
      msgs.insertBefore(div, bubble && bubble.parentNode ? bubble.parentNode : bubble);
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
  var bubble = document.getElementById('streamingBubble');
  if (bubble) { msgs.insertBefore(div, bubble && bubble.parentNode ? bubble.parentNode : bubble); }
  else { msgs.appendChild(div); }
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

  // Step type is nested: {type: "step", step: {type: "tool_call", ...}}
  var step = ev.step || ev;
  var stype = step.type || ev.step_type || '';
  var div = document.createElement('div');
  div.className = 'step ' + stype;

  // Insert before streamingBubble so final answer stays at bottom
  var bubble = document.getElementById('streamingBubble');

  if (stype === 'tool_call') {
    var tool = step.tool_name || '';
    var tagClass = U.toolTagClass(tool);
    var time = U.formatDuration(step.duration_ms);
    var argsStr = typeof step.tool_args === 'string' ? step.tool_args : JSON.stringify(step.tool_args||{}, null, 2);
    var resultStr = step.observation || '';

    div.innerHTML =
      '<div class="tool-header">'+
        '<span class="tool-tag '+tagClass+'">🔧 '+U.esc(tool)+'</span>'+
        (time ? '<span class="tool-time">⏱ '+time+'</span>' : '')+
      '</div>'+
      (argsStr && argsStr !== '{}' ?
        '<div class="tool-args" onclick="var c=this.querySelector(\'.tool-args-content\');c.classList.toggle(\'show\');var t=this.querySelector(\'.toggle\');t.textContent=c.classList.contains(\'show\')?\'▲ 收起\':\'▶ 展开\'">'+
        '参数 <span class="toggle">▲ 收起</span>'+
        '<div class="tool-args-content show"><pre style="margin:4px 0;font-size:11px">'+U.esc(argsStr.slice(0,500))+'</pre></div>'+
        '</div>' : '')+
      (resultStr ?
        '<div class="tool-result">'+U.renderMarkdown(resultStr.slice(0,500))+'</div>' : '');

    if (step.status === 'failed' || (resultStr && resultStr.indexOf('error') >= 0)) {
      div.style.borderColor = 'var(--danger)';
      div.style.background = '#FEF2F2';
    }
  } else if (stype === 'thought') {
    var thought = step.thought || '';
    var obs = step.observation || '';
    var content = thought || obs;
    div.innerHTML = '<strong>💡 思考</strong><div style="margin-top:4px">'+U.renderMarkdown(content.slice(0,500))+'</div>';
    if (thought && obs && obs !== thought) {
      div.innerHTML += '<div style="margin-top:6px;font-size:12px;color:var(--slate-5)">'+U.renderMarkdown(obs.slice(0,500))+'</div>';
    }
  } else if (stype === 'plan') {
    div.innerHTML = '<strong>📋 规划</strong><div style="margin-top:4px">'+U.renderMarkdown((step.plan||'').slice(0,500))+'</div>';
  } else if (stype === 'review') {
    div.className = 'step review';
    var parts = [];
    if (step.missing_steps && step.missing_steps.length) parts.push('<span style="color:var(--danger)">⚠ 缺失步骤: '+step.missing_steps.length+'</span>');
    if (step.flawed_logic && step.flawed_logic.length) parts.push('<span style="color:var(--warning)">⚠ 逻辑缺陷: '+step.flawed_logic.length+'</span>');
    if (step.boundary_gaps && step.boundary_gaps.length) parts.push('<span style="color:var(--info)">⚠ 边界缺口: '+step.boundary_gaps.length+'</span>');
    if (step.suggestions && step.suggestions.length) parts.push('<span>💡 建议: '+step.suggestions.length+' 条</span>');
    div.innerHTML = '<strong>🔍 审查结果</strong><div style="margin-top:4px">'+parts.join(' &middot; ')+'</div>';
  } else {
    var detailParts = ['<strong>'+U.esc(stype)+'</strong>'];
    if (step.thought) detailParts.push('<div style="margin-top:4px">'+U.renderMarkdown(String(step.thought).slice(0,500))+'</div>');
    if (step.observation) detailParts.push('<div style="margin-top:4px;font-size:12px;color:var(--slate-6)">'+U.renderMarkdown(String(step.observation).slice(0,500))+'</div>');
    if (step.tool_name) detailParts.push('<div style="margin-top:2px;font-size:11px;color:var(--slate-5)">工具: '+U.esc(step.tool_name)+'</div>');
    div.innerHTML = detailParts.join('');
  }

  // Insert before streamingBubble so answer stays at bottom
  if (bubble) {
    msgs.insertBefore(div, bubble && bubble.parentNode ? bubble.parentNode : bubble);
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
  var argsDisplay = ev.args || {};
  if (typeof argsDisplay === 'string') {
    try { argsDisplay = JSON.parse(argsDisplay); } catch(e) {}
  }
  document.getElementById('confirmArgs').textContent = typeof argsDisplay === 'object' ? JSON.stringify(argsDisplay, null, 2) : String(argsDisplay);
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
