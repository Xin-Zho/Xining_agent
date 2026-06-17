/* ============================================================
   Utils — Markdown, storage, helpers
   ============================================================ */
window.Utils = (function() {

// ── Storage ───────────────────────────────────────────
function skey(k) { return 'agent_'+(window.App&&App.username?App.username:localStorage.getItem('agent_user')||'default')+'_'+k; }
function lsave(k,v) { try{localStorage.setItem(skey(k),JSON.stringify(v))}catch(e){} }
function lload(k) { try{var v=localStorage.getItem(skey(k));return v?JSON.parse(v):null}catch(e){return null} }
function genId() { return 'x'+Date.now()+Math.random().toString(36).slice(2,8); }

// ── Toast ─────────────────────────────────────────────
function toast(msg, type) {
  type = type || 'info';
  var t = document.getElementById('toast');
  if (!t) { t = document.createElement('div'); t.id = 'toast'; document.body.appendChild(t); }
  t.className = 'toast '+type;
  t.textContent = msg;
  t.style.display = 'block';
  clearTimeout(t._t);
  t._t = setTimeout(function(){ t.style.display = 'none'; }, 4000);
}

// ── Tool tag color ────────────────────────────────────
function toolTagClass(name) {
  var n = (name||'').toLowerCase();
  if (/web|search|fetch|stock|rag/i.test(n)) return 'web';
  if (/command|shell|execute|bash/i.test(n)) return 'cmd';
  if (/excel|docx|document|create/i.test(n)) return 'doc';
  if (/file|read|grep|glob|edit/i.test(n)) return 'file';
  return 'other';
}

// ── Time formatting ────────────────────────────────────
function formatDuration(ms) {
  if (!ms || ms < 0) return '';
  if (ms < 1000) return ms+'ms';
  if (ms < 60000) return (ms/1000).toFixed(1)+'s';
  return Math.floor(ms/60000)+'m'+Math.round((ms%60000)/1000)+'s';
}

// ── HTML escape ────────────────────────────────────────
function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

// ── Markdown → HTML ───────────────────────────────────
function renderMarkdown(text) {
  if (!text) return '';
  var html = esc(text);

  // Code blocks with language tag + copy button
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, function(_, lang, code) {
    return '<pre><span class="code-lang">'+esc(lang||'')+'</span>'+
      '<button class="copy-btn" onclick="Utils.copyCode(this)">复制</button>'+
      '<code>'+esc(code.trimEnd())+'</code></pre>';
  });

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Tables
  html = html.replace(/((?:\|.+\|\n)+)/g, function(m) {
    var lines = m.trim().split('\n');
    if (lines.length < 2) return m;
    var html = '<table>';
    lines.forEach(function(line, i) {
      var cells = line.split('|').filter(function(c) { return c.trim(); });
      var tag = (i === 1 && /^[\s\-:]+$/.test(cells.join(''))) ? null : (i === 0 ? 'th' : 'td');
      if (!tag) return;
      html += '<tr>'+cells.map(function(c) { return '<'+tag+'>'+c.trim()+'</'+tag+'>'; }).join('')+'</tr>';
    });
    return html+'</table>';
  });

  // Headers
  html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

  // HR
  html = html.replace(/^---$/gm, '<hr>');

  // Lists
  html = html.replace(/^\- (.+)$/gm, '<li>$1</li>');
  html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

  // Links
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, function(_, text, url) {
    return '<a href="'+esc(url)+'" target="_blank" rel="noopener">'+esc(text)+'</a>';
  });

  // Bare URLs
  html = html.replace(/(?<!["'>])(https?:\/\/[^\s<>]+)/g, '<a href="$1" target="_blank" rel="noopener">$1</a>');

  // Line breaks
  html = html.replace(/\n/g, '<br>');

  return html;
}

// ── Code copy ──────────────────────────────────────────
function copyCode(btn) {
  var code = btn.parentNode.querySelector('code');
  if (!code) return;
  navigator.clipboard.writeText(code.textContent).then(function() {
    btn.textContent = '已复制';
    setTimeout(function(){ btn.textContent = '复制'; }, 1500);
  }).catch(function(){});
}

// ── Skeleton ───────────────────────────────────────────
function showSkeleton(container) {
  container.innerHTML = '<div class="skeleton text"></div><div class="skeleton text short"></div><div class="skeleton block"></div>';
}
function hideSkeleton(container) { container.innerHTML = ''; }

// ── Sending placeholder ────────────────────────────────
function addSendingPlaceholder() {
  var p = document.createElement('div');
  p.className = 'sending-placeholder';
  p.id = 'sendingPlaceholder';
  p.innerHTML = 'Agent 思考中 <span class="dot"></span><span class="dot"></span><span class="dot"></span>';
  var msgs = document.getElementById('messages');
  msgs.appendChild(p);
  scrollBottom();
}
function removeSendingPlaceholder() {
  var p = document.getElementById('sendingPlaceholder');
  if (p) p.remove();
}

// ── Scroll ─────────────────────────────────────────────
function scrollBottom() {
  var m = document.getElementById('messages');
  if (m) setTimeout(function(){ m.scrollTop = m.scrollHeight; }, 50);
}

// ── Token estimate ─────────────────────────────────────
function estTokens(text) { return Math.max(1, Math.ceil((text||'').length / 3)); }

return {
  skey:skey, lsave:lsave, lload:lload, genId:genId,
  toast:toast, toolTagClass:toolTagClass,
  formatDuration:formatDuration, esc:esc,
  renderMarkdown:renderMarkdown, copyCode:copyCode,
  showSkeleton:showSkeleton, hideSkeleton:hideSkeleton,
  addSendingPlaceholder:addSendingPlaceholder,
  removeSendingPlaceholder:removeSendingPlaceholder,
  scrollBottom:scrollBottom, estTokens:estTokens,
};
})();
