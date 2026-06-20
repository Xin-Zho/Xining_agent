/* ============================================================
   Auth — Login, Register, Token management
   ============================================================ */
window.Auth = (function() {
var U = Utils;

// ── State ──────────────────────────────────────────────
var isLoginMode = true;

function toggleAuthMode() {
  isLoginMode = !isLoginMode;
  document.getElementById('authTitle').textContent = isLoginMode ? '登录' : '注册';
  document.getElementById('authSubtitle').textContent = isLoginMode ? '欢迎回来' : '创建新账号';
  document.getElementById('authCodeLabel').style.display = isLoginMode ? 'none' : 'block';
  document.getElementById('authBtn').querySelector('.btn-text').textContent = isLoginMode ? '登 录' : '注 册';
  document.getElementById('switchText').textContent = isLoginMode ? '没有账号？' : '已有账号？';
  document.getElementById('switchLink').textContent = isLoginMode ? '注册' : '登录';
  document.getElementById('authError').textContent = '';
}

function showAuthPage() {
  document.getElementById('authPage').style.display = 'flex';
  document.getElementById('app').style.display = 'none';
  document.getElementById('authUser').value = '';
  document.getElementById('authPass').value = '';
  if (document.getElementById('authCode')) document.getElementById('authCode').value = '';
}

function showChatPage() {
  document.getElementById('authPage').style.display = 'none';
  document.getElementById('app').style.display = 'flex';
  document.getElementById('sidebarUser').textContent = App.username || '用户';
  App.renderSidebar();
  showChangelog();
}

// ── Auth ────────────────────────────────────────────────
async function handleAuth() {
  var user = document.getElementById('authUser').value.trim();
  var pass = document.getElementById('authPass').value;
  var code = document.getElementById('authCode') ? document.getElementById('authCode').value.trim() : '';
  var errEl = document.getElementById('authError');
  var btn = document.getElementById('authBtn');

  if (!user || !pass) { errEl.textContent = '请填写用户名和密码'; return; }
  if (!isLoginMode && pass.length < 8) { errEl.textContent = '密码至少 8 位'; return; }

  btn.classList.add('loading');
  btn.disabled = true;
  errEl.textContent = '';

  try {
    var resp = await fetch('/api/auth/' + (isLoginMode ? 'login' : 'register'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(isLoginMode ? { username: user, password: pass } : { username: user, password: pass, invite_code: code }),
    });
    var data = await resp.json();
    console.log('Auth response:', resp.status, data);

    if (resp.ok && (data.token || data.access_token)) {
      var token = data.token || data.access_token;
      App.userToken = token;
      App.username = user;
      localStorage.setItem('agent_token', token);
      localStorage.setItem('agent_user', user);
      showChatPage();
    } else {
      errEl.textContent = data.error || data.detail || (isLoginMode ? '登录失败 ('+resp.status+')' : '注册失败 ('+resp.status+')');
    }
  } catch (e) {
    console.error('Auth error:', e);
    errEl.textContent = '网络错误：'+e.message;
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

function logout() {
  App.userToken = '';
  App.username = '';
  App.currentConvId = null;
  App.conversations = [];
  localStorage.removeItem('agent_token');
  localStorage.removeItem('agent_user');
  showAuthPage();
}

// ── Changelog ───────────────────────────────────────────
function showChangelog() {
  if (sessionStorage.getItem('_clog')) return;
  sessionStorage.setItem('_clog', '1');
  var items = [
    '📊 新增 Agent 评估仪表盘 — 侧边栏点击 📊 查看',
    '💭 Agent 思考过程实时可见 — 流式展示 thinking 过程',
    '🔄 工具调用自动重试 — 超时后指数退避重试',
    '🔍 RAG 检索升级 — 语义+关键词混合检索',
    '📱 移动端体验优化 — 触控反馈 + 键盘适配',
  ];
  var html = '<strong>🆕 更新日志</strong><ul style="margin:8px 0 0;padding-left:18px;font-size:13px">';
  items.forEach(function(i) { html += '<li style="margin:4px 0">'+i+'</li>'; });
  html += '</ul>';
  document.getElementById('changelogContent').innerHTML = html;
  var toast = document.getElementById('changelogToast');
  toast.classList.add('show');
  setTimeout(function(){ toast.classList.remove('show'); }, 5000);
}

return { toggleAuthMode:toggleAuthMode, showAuthPage:showAuthPage, showChatPage:showChatPage,
         handleAuth:handleAuth, logout:logout };
})();
