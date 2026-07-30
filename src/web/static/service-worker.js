const CACHE_NAME = 'subpc-living-v30';
const STATIC_ASSETS = [
  '/',
  '/?source=pwa',
  '/tasks',
  '/logs',
  '/achievements',
  '/static/index.html',
  '/static/tasks.html',
  '/static/logs.html',
  '/static/achievements.html',
  '/static/tokens.css',
  '/static/style.css',
  '/static/hallmark-theme.css',
  '/static/fonts/geist-latin.woff2',
  '/static/fonts/instrument-serif-latin.woff2',
  '/static/fonts/jetbrains-mono-latin.woff2',
  '/static/fonts/LICENSES.txt',
  '/static/app.js',
  '/static/tasks.js',
  '/static/logs.js',
  '/static/achievements.js',
  '/static/shell-ui.js',
  '/static/favicon.svg',
  '/static/icon-192.png',
  '/static/icon-512.png',
  '/static/icon-maskable-512.png',
  '/static/manifest.json',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(
      keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
    ))
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const request = event.request;
  const url = new URL(request.url);
  if (request.method !== 'GET' || url.origin !== location.origin) return;
  if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/ws/')) return;

  // Query付きnavigationにはタスク名などが含まれ得るため、URLキーを永続保存しない。
  if (request.mode === 'navigate' && url.search) {
    event.respondWith(
      fetch(request).catch(() => caches.match(request).then((cached) => cached || caches.match('/')))
    );
    return;
  }

  event.respondWith(
    caches.match(request).then((cached) => {
      return cached || fetch(request).then((response) => {
        const copy = response.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(request, copy));
        return response;
      });
    })
  );
});
