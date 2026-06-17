/* ============================================================
   App — State, sidebar, conversations, agents, init
   ============================================================ */
window.App = (function() {
var U = Utils;

// ── State ───────────────────────────────────────────────
var userToken = '';
var username = '';
var conversations = [];
var currentConvId = null;
var agents = [];
var currentAgentId = null;
var selectedModel = 'chat';
var agentMode = 'react';

// ── Default agents ──────────────────────────────────────
function getDefaultAgents() {
  return [
    {id:'_builtin_stock', name:'股票分析师', description:'实时股票查询+Excel报表', systemPrompt:'你是一个专业股票分析师。用stock_query工具获取数据，用create_excel生成报表。简洁准确。'},
    {id:'_builtin_coder', name:'代码助手', description:'代码编写/调试/重构', systemPrompt:'你是高级软件工程师。用grep_files/file_read理解代码，用edit_file精确修改。先理解再动手。'},
    {id:'_builtin_writer', name:'文档生成器', description:'PDF阅读+报告生成', systemPrompt:'你是专业文档专家。用read_pdf提取内容，用create_docx/create_excel生成文档。格式规范。'},
    {id:'_builtin_researcher', name:'深度研究员', description:'多源搜索+分析', systemPrompt:'你是研究分析师。用web_search发现信息，用web_fetch深度阅读，交叉验证后给出结论。标注信息来源。'},
  ];
}

// ── Storage ─────────────────────────────────────────────
function loadConvs() {
  conversations = U.lload('convs') || [];
}
function saveConvs() {
  U.lsave('convs', conversations.slice(0, 50));
}
function loadAgents() {
  agents = U.lload('agents') || [];
  // Merge builtins
  var builtins = getDefaultAgents();
  builtins.forEach(function(b) {
    if (!agents.find(function(a){ return a.id===b.id; })) {
      agents.unshift(b);
    }
  });
}
function saveAgents() {
  var custom = agents.filter(function(a){ return !String(a.id).startsWith('_builtin_'); });
  U.lsave('agents', custom);
}

// ── Sidebar ─────────────────────────────────────────────
function renderSidebar() {
  // Conversations
  var el = document.getElementById('convList');
  el.innerHTML = '';
  var sorted = (conversations||[]).slice().sort(function(a,b){ return (b.updatedAt||0)-(a.updatedAt||0); });
  sorted.forEach(function(conv) {
    var div = document.createElement('div');
    div.className = 'conv-item' + (conv.id===currentConvId?' active':'');
    div.innerHTML = '<span class="title">'+U.esc(conv.title||'新对话')+'</span>'+
      '<button class="del-btn" onclick="event.stopPropagation();App.deleteConv(\''+conv.id+'\')">×</button>';
    div.onclick = function(){ App.switchConv(conv.id); };
    el.appendChild(div);
  });

  // Agents
  var ael = document.getElementById('agentList');
  ael.innerHTML = '';
  agents.forEach(function(a) {
    var div = document.createElement('div');
    div.className = 'agent-item' + (a.id===currentAgentId?' active':'');
    div.innerHTML = '<span>🤖</span><span class="title">'+U.esc(a.name)+'</span>';
    if (!String(a.id).startsWith('_builtin_')) {
      div.innerHTML += '<button class="del-btn" onclick="event.stopPropagation();App.deleteAgent(\''+a.id+'\')">×</button>';
    }
    div.onclick = function(){ App.selectAgent(a.id); };
    ael.appendChild(div);
  });
}

function selectAgent(id) {
  currentAgentId = id;
  var badge = document.getElementById('agentBadge');
  var sub = document.getElementById('headerSub');
  if (id) {
    var a = agents.find(function(x){ return x.id===id; });
    badge.style.display = 'inline-block';
    badge.textContent = a ? a.name : 'Agent';
    sub.textContent = a ? a.description : '';
  } else {
    badge.style.display = 'none';
    sub.textContent = '';
  }
  renderSidebar();
}

function switchConv(id) {
  currentConvId = id;
  Chat.renderMessages();
  renderSidebar();
}

function newConversation() {
  var conv = { id: U.genId(), title: '新对话', messages: [], updatedAt: Date.now() };
  conversations.push(conv);
  currentConvId = conv.id;
  saveConvs();
  renderSidebar();
  Chat.renderMessages();
}

function deleteConv(id) {
  conversations = conversations.filter(function(c){ return c.id!==id; });
  if (currentConvId === id) {
    currentConvId = conversations.length ? conversations[conversations.length-1].id : null;
  }
  saveConvs();
  renderSidebar();
  Chat.renderMessages();
}

// ── Agent CRUD ──────────────────────────────────────────
function openAgentModal(editId) {
  var modal = document.getElementById('agentModalOverlay');
  modal.classList.add('show');
  if (editId) {
    var a = agents.find(function(x){ return x.id===editId; });
    if (a) {
      document.getElementById('agentModalTitle').textContent = '编辑 Agent';
      document.getElementById('agentName').value = a.name;
      document.getElementById('agentDesc').value = a.description||'';
      document.getElementById('agentPrompt').value = a.systemPrompt||'';
      document.getElementById('agentEditId').value = editId;
    }
  } else {
    document.getElementById('agentModalTitle').textContent = '创建 Agent';
    document.getElementById('agentName').value = '';
    document.getElementById('agentDesc').value = '';
    document.getElementById('agentPrompt').value = '';
    document.getElementById('agentEditId').value = '';
  }
}

function closeAgentModal() {
  document.getElementById('agentModalOverlay').classList.remove('show');
}

function saveAgent() {
  var editId = document.getElementById('agentEditId').value;
  var name = document.getElementById('agentName').value.trim();
  var desc = document.getElementById('agentDesc').value.trim();
  var prompt = document.getElementById('agentPrompt').value.trim();

  if (!name) { U.toast('请输入名称','warn'); return; }

  if (editId) {
    var a = agents.find(function(x){ return x.id===editId; });
    if (a) { a.name=name; a.description=desc; a.systemPrompt=prompt; }
  } else {
    agents.push({id:U.genId(), name:name, description:desc, systemPrompt:prompt});
  }

  saveAgents();
  closeAgentModal();
  renderSidebar();
  if (!currentAgentId || editId===currentAgentId) selectAgent(editId||agents[agents.length-1].id);
}

function deleteAgent(id) {
  if (String(id).startsWith('_builtin_')) { U.toast('内置 Agent 不可删除','warn'); return; }
  agents = agents.filter(function(a){ return a.id!==id; });
  if (currentAgentId === id) selectAgent(null);
  saveAgents();
  renderSidebar();
}

// ── Settings ────────────────────────────────────────────
function onModelChange() {
  selectedModel = document.getElementById('modelSelect').value;
  var badge = document.getElementById('modelBadge');
  badge.style.display = 'inline-block';
  badge.textContent = selectedModel === 'reasoner' ? '深度' : '快速';
}

function onFontSizeChange() {
  var v = document.getElementById('fontSizeSelect').value;
  document.body.classList.toggle('text-large', v === 'large');
  localStorage.setItem('agent_font', v);
}

function onAgentModeChange() {
  agentMode = document.getElementById('modeSelect').value;
}

// ── Sidebar toggle ──────────────────────────────────────
function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('open');
  document.getElementById('overlay').classList.toggle('show');
}
function closeSidebar() {
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('overlay').classList.remove('show');
}

// ── Init ────────────────────────────────────────────────
function init() {
  userToken = localStorage.getItem('agent_token') || '';
  username = localStorage.getItem('agent_user') || '';
  var font = localStorage.getItem('agent_font');
  if (font) { document.getElementById('fontSizeSelect').value = font; document.body.classList.toggle('text-large', font==='large'); }

  if (userToken) {
    fetch('/api/auth/me', {headers:{'Authorization':'Bearer '+userToken}})
      .then(function(r){ return r.json(); })
      .then(function(data) {
        if (data.username) {
          username = data.username;
          loadConvs();
          loadAgents();
          Auth.showChatPage();
        } else {
          Auth.logout();
        }
      })
      .catch(function(){ Auth.logout(); });
  } else {
    Auth.showAuthPage();
  }
}

// Overlay clicks
document.addEventListener('DOMContentLoaded', function() {
  document.getElementById('overlay').addEventListener('click', closeSidebar);
});

init();

return {
  get userToken() { return userToken; }, set userToken(v) { userToken = v; },
  get username() { return username; }, set username(v) { username = v; },
  get conversations() { return conversations; }, set conversations(v) { conversations = v; },
  get currentConvId() { return currentConvId; }, set currentConvId(v) { currentConvId = v; },
  get agents() { return agents; }, set agents(v) { agents = v; },
  get currentAgentId() { return currentAgentId; },
  get selectedModel() { return selectedModel; },
  get agentMode() { return agentMode; },
  renderSidebar:renderSidebar, selectAgent:selectAgent,
  switchConv:switchConv, newConversation:newConversation, deleteConv:deleteConv,
  openAgentModal:openAgentModal, closeAgentModal:closeAgentModal,
  saveAgent:saveAgent, deleteAgent:deleteAgent,
  onModelChange:onModelChange, onFontSizeChange:onFontSizeChange,
  onAgentModeChange:onAgentModeChange,
  toggleSidebar:toggleSidebar, closeSidebar:closeSidebar,
  loadConvs:loadConvs, saveConvs:saveConvs,
};
})();
