/**
 * Formats an unknown error value into a human-readable string.
 * Used by command handlers to produce consistent error messages.
 */
export function formatError(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  if (typeof error === 'string') {
    return error;
  }
  return String(error);
}

/**
 * Generates an HTML representation of a report for the webview.
 */
export function generateReportHtml(report: unknown): string {
  const reportContent = typeof report === 'string' ? report : JSON.stringify(report, null, 2);
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'none';">
  <title>Report</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      line-height: 1.6;
      color: #333;
      max-width: 900px;
      margin: 0 auto;
      padding: 16px;
    }
    h1 { color: #1a1a1a; border-bottom: 2px solid #007acc; padding-bottom: 8px; }
    h2 { color: #2d2d2d; margin-top: 24px; }
    pre { background: #f6f8fa; padding: 12px; border-radius: 6px; overflow-x: auto; }
    .section { margin-bottom: 24px; }
  </style>
</head>
<body>
  <h1>Analysis Report</h1>
  <div class="section">
    <pre>${escapeHtml(reportContent)}</pre>
  </div>
</body>
</html>`;
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}