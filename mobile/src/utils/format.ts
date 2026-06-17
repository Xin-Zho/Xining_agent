export function formatDate(dateStr: string): string {
  const d = new Date(dateStr + 'Z');
  const now = new Date();
  const diff = now.getTime() - d.getTime();
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);

  if (minutes < 1) return '刚刚';
  if (minutes < 60) return `${minutes}分钟前`;
  if (hours < 24) return `${hours}小时前`;
  if (days < 7) return `${days}天前`;
  return d.toLocaleDateString('zh-CN');
}

export function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
}

export function statusLabel(status: string): string {
  const map: Record<string, string> = {
    pending: '等待中',
    planning: '规划中',
    executing: '执行中',
    observing: '观察中',
    completed: '已完成',
    failed: '失败',
    cancelled: '已取消',
  };
  return map[status] || status;
}

export function statusColor(status: string): string {
  const map: Record<string, string> = {
    pending: '#888',
    planning: '#2196F3',
    executing: '#FF9800',
    observing: '#9C27B0',
    completed: '#4CAF50',
    failed: '#F44336',
    cancelled: '#888',
  };
  return map[status] || '#888';
}
