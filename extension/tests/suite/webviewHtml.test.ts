import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import * as vm from 'vm';
import {
  buildChatHtml,
  buildSettingsHtml,
  buildContentSecurityPolicy,
  escapeHtml,
  createNonce,
} from '../../src/views/webviewHtml';
import { PROVIDERS } from '../../src/services/providers';

const MEDIA_DIR = path.resolve(__dirname, '..', '..', 'media');
const assets = { cspSource: 'vscode-webview://abc', nonce: 'NONCE123', scriptUri: 'vscode-webview://abc/media/x.js' };

describe('media scripts', () => {
  // This is the regression that left every button dead: the script did not
  // parse. Parsing the real files (not a string inside TypeScript) catches it.
  for (const file of ['chat.js', 'settings.js']) {
    it(`${file} is syntactically valid JavaScript`, () => {
      const source = fs.readFileSync(path.join(MEDIA_DIR, file), 'utf8');
      expect(() => new vm.Script(source, { filename: file })).not.toThrow();
    });

    it(`${file} never assigns to innerHTML`, () => {
      const source = fs.readFileSync(path.join(MEDIA_DIR, file), 'utf8');
      expect(source).not.toMatch(/\.innerHTML\s*=/);
    });
  }
});

describe('content security policy', () => {
  it('allows only the nonce-tagged script and has no unsafe-inline script', () => {
    const csp = buildContentSecurityPolicy('vscode-webview://abc', 'N1');
    expect(csp).toContain("default-src 'none'");
    expect(csp).toContain("script-src 'nonce-N1'");
    expect(csp).not.toMatch(/script-src[^;]*unsafe-inline/);
    expect(csp).toContain('worker-src vscode-webview://abc');
  });

  it('creates unpredictable nonces', () => {
    expect(createNonce()).not.toBe(createNonce());
    expect(createNonce()).toMatch(/^[0-9a-f]{32}$/);
  });
});

describe('chat html', () => {
  const html = buildChatHtml(assets);

  it('has the controls the script binds to and one external nonce script', () => {
    expect(html).toContain('id="send"');
    expect(html).toContain('id="input"');
    expect(html).toContain('id="messages"');
    const scripts = html.match(/<script[^>]*>/g) ?? [];
    expect(scripts).toEqual([`<script nonce="NONCE123" src="vscode-webview://abc/media/x.js">`]);
    expect(html).not.toMatch(/<script[^>]*>[^<]+<\/script>/);
  });
});

describe('settings html', () => {
  const options = {
    ...assets,
    provider: 'openai',
    model: 'gpt-4o-mini',
    keyStates: { openai: true, anthropic: false },
    baseUrls: { ollama: 'http://localhost:11434' },
  };

  it('lists every provider and selects the active one', () => {
    const html = buildSettingsHtml(options);
    for (const provider of PROVIDERS) {
      expect(html).toContain(`value="${provider.id}"`);
    }
    expect(html).toContain('value="openai" selected');
  });

  it('passes state through escaped data attributes, never into script code', () => {
    const html = buildSettingsHtml({ ...options, provider: 'x"><script>alert(1)</script>', model: `o'brien "q" <b>` });
    expect(html).not.toContain('<script>alert(1)</script>');
    expect(html).toContain('data-model="o&#39;brien &quot;q&quot; &lt;b&gt;"');
    expect((html.match(/<script/g) ?? []).length).toBe(1);
  });

  it('only reports whether a key is saved', () => {
    const html = buildSettingsHtml(options);
    expect(html).toContain('data-key-states="{&quot;openai&quot;:true,&quot;anthropic&quot;:false}"');
  });

  it('escapeHtml neutralises markup characters', () => {
    expect(escapeHtml(`<a href="x">&'</a>`)).toBe('&lt;a href=&quot;x&quot;&gt;&amp;&#39;&lt;/a&gt;');
  });
});
