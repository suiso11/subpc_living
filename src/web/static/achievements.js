const achievementState = {
  badges: [],
  filter: 'all',
};

const achievementEl = (selector) => document.querySelector(selector);

function formatAchievementCount(value) {
  return Number(value || 0).toLocaleString('ja-JP');
}

function achievementProgressText(badge) {
  const current = Math.min(Number(badge.current || 0), Number(badge.target || 0));
  if (badge.unit === 'Lv') return `Lv.${current} / Lv.${badge.target}`;
  return `${current} / ${badge.target} ${badge.unit || ''}`.trim();
}

function createAchievementCard(badge) {
  const card = document.createElement('article');
  card.className = `achievement-card ${badge.unlocked ? 'unlocked' : 'locked'}`;
  card.dataset.state = badge.unlocked ? 'unlocked' : 'locked';

  const top = document.createElement('div');
  top.className = 'achievement-card-top';
  const mark = document.createElement('span');
  mark.className = 'achievement-card-mark';
  mark.textContent = badge.unlocked ? badge.mark : '？';
  mark.setAttribute('aria-hidden', 'true');
  const status = document.createElement('span');
  status.className = 'achievement-card-status';
  status.textContent = badge.unlocked ? '解除済み' : '挑戦中';
  top.append(mark, status);

  const name = document.createElement('h3');
  name.textContent = badge.name;
  const detail = document.createElement('p');
  detail.textContent = badge.detail;

  const progressText = achievementProgressText(badge);
  const progress = document.createElement('div');
  progress.className = 'achievement-card-progress';
  progress.setAttribute('role', 'progressbar');
  progress.setAttribute('aria-label', `${badge.name}：${progressText}`);
  progress.setAttribute('aria-valuemin', '0');
  progress.setAttribute('aria-valuemax', String(badge.target));
  progress.setAttribute('aria-valuenow', String(Math.min(badge.current, badge.target)));
  const bar = document.createElement('span');
  bar.style.width = `${Math.min(100, Number(badge.current || 0) / Math.max(1, Number(badge.target || 1)) * 100)}%`;
  progress.appendChild(bar);
  const count = document.createElement('strong');
  count.className = 'achievement-card-count';
  count.textContent = progressText;

  card.append(top, name, detail, progress, count);
  return card;
}

function renderAchievementGrid() {
  const visible = achievementState.badges.filter((badge) => (
    achievementState.filter === 'all'
    || (achievementState.filter === 'unlocked' && badge.unlocked)
    || (achievementState.filter === 'locked' && !badge.unlocked)
  ));
  achievementEl('#achievement-grid').replaceChildren(...visible.map(createAchievementCard));
  achievementEl('#achievement-empty').hidden = visible.length !== 0;
}

function renderAchievements(data) {
  const badges = Array.isArray(data.badges) ? data.badges : [];
  achievementState.badges = badges;

  achievementEl('#achievement-rank-name').textContent = data.rank.name;
  achievementEl('#achievement-next-rank').textContent = data.rank.next
    ? `次は Lv.${data.rank.next.level}「${data.rank.next.name}」`
    : '最高ランクに到達しています';
  achievementEl('#achievement-points').textContent = formatAchievementCount(data.points);
  achievementEl('#achievement-unlocked').textContent = formatAchievementCount(data.unlocked_badges);
  achievementEl('#achievement-total').textContent = `/ ${badges.length}`;
  achievementEl('#achievement-overview').hidden = false;

  const locked = badges
    .filter((badge) => !badge.unlocked)
    .sort((a, b) => (b.current / b.target) - (a.current / a.target));
  const next = locked[0];
  const nextSection = achievementEl('#achievement-next');
  if (next) {
    achievementEl('#achievement-next-title').textContent = next.name;
    achievementEl('#achievement-next-detail').textContent = next.detail;
    achievementEl('#achievement-next-progress').textContent = achievementProgressText(next);
    nextSection.hidden = false;
  } else {
    nextSection.hidden = true;
  }
  renderAchievementGrid();
}

async function loadAchievements() {
  const error = achievementEl('#achievement-error');
  error.textContent = '';
  try {
    const response = await fetch('/api/game', { cache: 'no-store' });
    if (!response.ok) throw new Error(`game ${response.status}`);
    const data = await response.json();
    if (!data.enabled) throw new Error('game unavailable');
    renderAchievements(data);
  } catch (caught) {
    error.textContent = '実績を読み込めませんでした。画面を読み直してください。';
    console.warn('[Achievements] Fetch failed:', caught);
  }
}

document.querySelectorAll('.achievement-filter-btn').forEach((button) => {
  button.addEventListener('click', () => {
    achievementState.filter = button.dataset.filter;
    document.querySelectorAll('.achievement-filter-btn').forEach((item) => {
      const active = item === button;
      item.classList.toggle('active', active);
      item.setAttribute('aria-pressed', String(active));
    });
    renderAchievementGrid();
  });
});

loadAchievements();
