import { describe, it, expect } from 'vitest';
import { formatError, generateReportHtml } from '../../src/utils';

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