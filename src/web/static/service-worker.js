const CACHE_NAME = 'subpc-living-v22';
const STATIC_ASSETS = [
  '/',
  '/tasks',
  '/logs',
  '/achievements',
  '/static/index.html',
  '/static/tasks.html',
  '/static/logs.html',
  '/static/achievements.html',
  '/static/style.css',
  '/static/pop-theme.css',
  '/static/shell-theme.css',
  '/static/app.js',
  '/static/tasks.js',
  '/static/logs.js',
  '/static/achievements.js',
  '/static/shell-ui.js',
  '/static/favicon.svg',
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
