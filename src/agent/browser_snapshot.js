// Runs in a CDP isolated world. Page scripts cannot replace our element map.
(() => {
  const clean = (s, max = 200) => String(s || '').replace(/\s+/g, ' ').trim().slice(0, max);
  const visible = (el) => {
    const style = getComputedStyle(el);
    return style.display !== 'none' && style.visibility !== 'hidden' &&
      style.visibility !== 'collapse' && Number(style.opacity) !== 0 &&
      !el.closest('[hidden],[inert],[aria-hidden="true"]') && el.getClientRects().length > 0;
  };
  const label = (el) => clean(el.getAttribute('aria-label') ||
    [...(el.labels || [])].map(l => l.innerText).join(' ') ||
    el.innerText || el.getAttribute('alt') || el.getAttribute('placeholder') ||
    el.getAttribute('title') || el.getAttribute('name'));
  const fingerprint = (el) => JSON.stringify([
    el.tagName, el.getAttribute('type'), el.getAttribute('role'), label(el),
    el.getAttribute('href'), el.getAttribute('formaction'), el.form?.getAttribute('action'),
    el.disabled, el.readOnly,
  ]);
  const secret = (el) => el.matches('input[type="password"]') ||
    /password|one-time-code|cc-number|cc-csc/.test(el.getAttribute('autocomplete') || '');
  const elements = [];
  const refs = [];
  const viewport = (rect) => rect.width > 0 && rect.height > 0 &&
    rect.bottom > 0 && rect.right > 0 && rect.top < innerHeight && rect.left < innerWidth;
  const textParts = [];
  let textLength = 0;
  let textTruncated = false;
  const collectText = (root) => {
    const walker = document.createTreeWalker(root === document ? document.body : root, NodeFilter.SHOW_TEXT);
    let node, scanned = 0;
    while ((node = walker.nextNode())) {
      if (++scanned > 10000) { textTruncated = true; break; }
      const parent = node.parentElement || root.host;
      if (!parent || !node.textContent.trim() || !visible(parent) ||
          parent.closest('script,style,noscript')) continue;
      const range = document.createRange();
      range.selectNodeContents(node);
      if (![...range.getClientRects()].some(viewport)) continue;
      const value = clean(node.textContent, 12000);
      if (textLength + value.length > 12000) { textTruncated = true; break; }
      textParts.push(value); textLength += value.length + 1;
    }
  };
  let omitted = false;
  const collect = (root) => {
    collectText(root);
    for (const el of root.querySelectorAll('*')) {
      if (!visible(el)) continue;
      if (el.shadowRoot) collect(el.shadowRoot);
      if (!viewport(el.getBoundingClientRect())) continue;
      if (!el.matches('a[href],button,input:not([type="hidden"]),textarea,select,' +
        '[role="button"],[role="link"],[role="textbox"],[contenteditable="true"],summary')) continue;
      if (refs.length >= 150) { omitted = true; continue; }
      const id = refs.length + 1;
      refs.push({ el, fingerprint: fingerprint(el) });
      elements.push({
        id, tag: el.tagName.toLowerCase(), role: el.getAttribute('role') || undefined,
        type: el.getAttribute('type') || undefined, label: label(el),
        href: el.tagName === 'A' ? clean(el.getAttribute('href'), 500) : undefined,
        disabled: el.disabled || undefined,
        value: !secret(el) && /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName)
          ? clean(el.value, 300) : undefined,
        checked: el.matches('input[type="checkbox"],input[type="radio"]') ? el.checked : undefined,
        options: el.tagName === 'SELECT' ? [...el.options].slice(0, 60).map(o => ({
          value: clean(o.value), label: clean(o.text), selected: o.selected,
        })) : undefined,
      });
    }
  };
  collect(document);
  globalThis.__kotoniaBrowser = {
    token: SNAPSHOT_TOKEN, refs, fingerprint, visible, url: location.href,
  };
  return {
    snapshot: SNAPSHOT_TOKEN, url: location.href, title: clean(document.title, 300),
    text: textParts.join('\n'), text_truncated: textTruncated,
    scroll: { x: scrollX, y: scrollY, height: document.documentElement.scrollHeight, viewport_height: innerHeight },
    elements, elements_truncated: omitted,
    frames: [...document.querySelectorAll('iframe')].filter(el => visible(el) && viewport(el.getBoundingClientRect())).map(el => ({
      title: clean(el.title), src: clean(el.getAttribute('src'), 500),
    })).slice(0, 20),
  };
})()
