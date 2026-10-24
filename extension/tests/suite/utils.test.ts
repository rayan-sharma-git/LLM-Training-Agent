import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import { formatError, generateReportHtml, escapeJsString } from '../../src/utils';

describe('formatError', () => {
  it('should format Error instances', () => {
    expect(formatError(new Error('broken'))).toBe('broken');
  });

  it('should format strings', () => {
    expect(formatError('string error')).toBe('string error');
  });

  it('should coerce unknown values to string', () => {
    expect(formatError(123)).toBe('123');
    expect(formatError(null)).toBe('null');
  });
});

describe('generateReportHtml', () => {
  it('should generate basic HTML for object reports', () => {
    const report = { summary: 'ok', items: [1, 2] };
    const html = generateReportHtml(report);
    expect(html).toContain('<!DOCTYPE html>');
    expect(html).toContain('Analysis Report');
    // Quotes are HTML-escaped in the output.
    expect(html).toContain('\u0026quot;summary\u0026quot;: \u0026quot;ok\u0026quot;');
    expect(html).toContain('<pre>');
  });

  it('should escape HTML in string reports', () => {
    const html = generateReportHtml('<b>raw</b>');
    // The escaped output must contain the HTML entities, not the raw tags.
    expect(html).toContain('\u0026lt;b\u0026gt;raw\u0026lt;/b\u0026gt;');
    expect(html).not.toContain('<b>raw</b>');
  });
});

describe('escapeJsString', () => {
  it('escapes single quotes so interpolated values cannot end the literal', () => {
    expect(escapeJsString("o'brien")).toBe("o\\'brien");
  });

  it('escapes backslashes, newlines, angle brackets and backticks', () => {
    expect(escapeJsString('a\\b')).toBe('a\\\\b');
    expect(escapeJsString('x\ny')).toBe('x\\ny');
    expect(escapeJsString('x\ry')).toBe('x\\ry');
    expect(escapeJsString('</script>')).toBe('\\x3c/script\\x3e');
    expect(escapeJsString('a`b')).toBe('a\\x60b');
  });

  it('round-trips through evaluation to the original value', () => {
    // The escaped payload is placed inside '...' in the webview script, so
    // evaluating the literal must reproduce the input exactly.
    for (const value of ["o'brien", 'a\\b', 'line1\nline2', '</script>', 'plain-llama3.2']) {
      // eslint-disable-next-line no-eval
      expect(eval(`'${escapeJsString(value)}'`)).toBe(value);
    }
  });
});
