/* ============================================================
   Eval — Evaluation dashboard (总览 / 弱点 / 建议)
   ============================================================ */
window.Eval = (function() {
var U = Utils;

var currentTab = 'overview';

function open() {
  document.getElementById('main').style.display = 'none';
  document.getElementById('evalPage').classList.add('show');
  loadTab('overview');
}

function close() {
  document.getElementById('evalPage').classList.remove('show');
  document.getElementById('main').style.display = 'flex';
}

function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.eval-tab').forEach(function(t){ t.classList.toggle('active', t.textContent.includes(tab==='overview'?'总览':tab==='weaknesses'?'弱点':'建议')); });
  loadTab(tab);
}

async function loadTab(tab) {
  var body = document.getElementById(tab === 'overview' ? 'evalOverview' : tab === 'weaknesses' ? 'evalWeaknesses' : 'evalRecommendations');
  document.querySelectorAll('.eval-body').forEach(function(b){ b.hidden = true; });
  body.hidden = false;

  U.showSkeleton(body);

  try {
    switch(tab) {
      case 'overview': await loadOverview(body); break;
      case 'weaknesses': await loadWeaknesses(body); break;
      case 'recommendations': await loadRecommendations(body); break;
    }
  } catch(e) {
    body.innerHTML = '<div style="color:var(--danger);padding:20px;text-align:center">加载失败: '+U.esc(e.message)+'</div>';
  }
}

// ── Overview ────────────────────────────────────────────
async function loadOverview(el) {
  var resp = await fetch('/api/evaluation/dashboard', {
    headers: {'Authorization':'Bearer '+App.userToken}
  });
  var d = await resp.json();

  var html = '<div class="eval-grid">';

  // Success rate
  var srPct = Math.round(d.success_rate*100);
  html += card('成功率', srPct+'%', srPct >= 90 ? 'good' : srPct >= 70 ? 'warn' : 'bad');
  html += card('总任务', d.total_tasks||0, '');
  html += card('平均质量分', d.avg_quality_score ? d.avg_quality_score.toFixed(1)+'/10' : 'N/A', d.avg_quality_score >= 7 ? 'good' : 'warn');
  html += card('工具错误率', d.avg_tool_error_rate ? Math.round(d.avg_tool_error_rate*100)+'%' : '0%', d.avg_tool_error_rate < 0.1 ? 'good' : 'warn');
  html += card('平均耗时', U.formatDuration(d.avg_duration_ms)||'N/A', '');
  html += card('Token 效率', d.avg_token_efficiency ? d.avg_token_efficiency.toFixed(4) : 'N/A', '');

  html += '</div>';

  // Mode comparison
  if (d.by_mode && Object.keys(d.by_mode).length > 1) {
    html += '<h3 style="margin:16px 0 8px">模式对比</h3><div class="eval-mode-compare">';
    Object.keys(d.by_mode).forEach(function(mode) {
      var m = d.by_mode[mode];
      html += '<div class="mode-col"><div class="mode-name">'+U.esc(mode)+'</div>';
      html += '<div class="mode-sr" style="color:'+(m.success_rate>=0.9?'var(--success)':'var(--warning)')+'">'+Math.round(m.success_rate*100)+'%</div>';
      html += '<div style="font-size:12px;color:var(--slate-5)">'+m.count+' 任务</div></div>';
    });
    html += '</div>';
  }

  // Tool ranking
  if (d.top_failing_tools && d.top_failing_tools.length) {
    html += '<h3 style="margin:16px 0 8px">工具调用排行</h3>';
    var maxErr = Math.max.apply(null, d.top_failing_tools.map(function(t){ return t.error_rate; }));
    d.top_failing_tools.forEach(function(t,i) {
      var barW = maxErr > 0 ? Math.round(t.error_rate/maxErr*100) : 0;
      html += '<div class="tool-rank-item">'+
        '<span class="rank">'+(i+1)+'</span>'+
        '<span class="name">'+U.esc(t.tool_name)+'</span>'+
        '<span style="font-size:12px;width:60px;text-align:right">'+t.total_calls+'次</span>'+
        '<div class="bar"><div class="bar-fill" style="width:'+barW+'%"></div></div>'+
        '<span style="font-size:12px;width:50px;text-align:right">'+(t.error_rate*100).toFixed(1)+'%</span>'+
      '</div>';
    });
  }

  el.innerHTML = html;
}

// ── Weaknesses ──────────────────────────────────────────
async function loadWeaknesses(el) {
  var resp = await fetch('/api/evaluation/weaknesses', {
    headers: {'Authorization':'Bearer '+App.userToken}
  });
  var d = await resp.json();

  var html = '';
  if (d.summary) {
    html += '<div style="background:var(--slate-1);padding:14px;border-radius:var(--radius-md);margin-bottom:16px;font-size:14px">'+U.esc(d.summary)+'</div>';
  }

  if (!d.weaknesses || !d.weaknesses.length) {
    html += '<div style="text-align:center;color:var(--slate-5);padding:40px">✅ 未检测到明显弱项</div>';
  } else {
    d.weaknesses.forEach(function(w) {
      html += '<div class="weakness-card '+w.severity+'">'+
        '<div class="title">'+U.esc(w.description)+'</div>'+
        '<div class="desc">'+U.esc(w.evidence||'')+'</div>'+
      '</div>';
    });
  }

  el.innerHTML = html;
}

// ── Recommendations ─────────────────────────────────────
async function loadRecommendations(el) {
  // Auto-generate fresh recommendations
  try {
    await fetch('/api/evaluation/recommendations/generate', {
      method:'POST',
      headers: {'Authorization':'Bearer '+App.userToken}
    });
  } catch(e) {}

  var resp = await fetch('/api/evaluation/recommendations', {
    headers: {'Authorization':'Bearer '+App.userToken}
  });
  var d = await resp.json();

  var html = '<div style="margin-bottom:16px;font-size:14px;color:var(--slate-6)">'+U.esc(d.summary||'')+'</div>';

  if (!d.recommendations || !d.recommendations.length) {
    html += '<div style="text-align:center;color:var(--slate-5);padding:40px">✅ 暂无待处理建议</div>';
  } else {
    d.recommendations.forEach(function(r) {
      var sc = r.suggested_change || {};
      html += '<div class="rec-card '+r.severity+'">'+
        '<div class="title">'+U.esc(r.title)+'</div>'+
        '<div class="desc">'+U.esc(r.description)+'<br>'+
          '<span style="color:var(--slate-5);font-size:12px">当前: '+U.esc(sc.current||'')+' → 建议: '+U.esc(sc.proposed||'')+'</span>'+
        '</div>';


      html += '</div>';
    });
  }

  el.innerHTML = html;
}

// ── Helper ──────────────────────────────────────────────
function card(label, value, cls) {
  return '<div class="eval-card"><div class="label">'+U.esc(label)+'</div><div class="value'+(cls?' '+cls:'')+'">'+U.esc(String(value))+'</div></div>';
}

return {
  open:open, close:close, switchTab:switchTab,
  loadTab:loadTab,
};
})();
