import * as vscode from 'vscode';

/**
 * A simple tree item representing a single node in a sidebar view.
 */
export class SimpleTreeItem extends vscode.TreeItem {
  constructor(
    public readonly label: string,
    public readonly collapsibleState: vscode.TreeItemCollapsibleState,
    public readonly command?: vscode.Command
  ) {
    super(label, collapsibleState);
  }
}

/**
 * Minimal TreeDataProvider that displays a list of string entries.
 * Used by the Overview and Reports sidebar views.
 */
export class SimpleTreeDataProvider implements vscode.TreeDataProvider<SimpleTreeItem> {
  private _onDidChangeTreeData = new vscode.EventEmitter<SimpleTreeItem | undefined | void>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  constructor(private entries: { label: string; command?: vscode.Command }[] = []) {}

  refresh(entries: { label: string; command?: vscode.Command }[]): void {
    this.entries = entries;
    this._onDidChangeTreeData.fire();
  }

  getTreeItem(element: SimpleTreeItem): vscode.TreeItem {
    return element;
  }

  getChildren(): vscode.ProviderResult<SimpleTreeItem[]> {
    if (this.entries.length === 0) {
      return [new SimpleTreeItem('No data available', vscode.TreeItemCollapsibleState.None)];
    }
    return this.entries.map(
      (entry) =>
        new SimpleTreeItem(entry.label, vscode.TreeItemCollapsibleState.None, entry.command)
    );
  }
}

/**
 * Creates and registers a TreeDataProvider for a given view ID.
 * Returns the provider so the caller can refresh it later.
 */
export function registerTreeView(
  context: vscode.ExtensionContext,
  viewId: string,
  entries: { label: string; command?: vscode.Command }[] = []
): SimpleTreeDataProvider {
  const provider = new SimpleTreeDataProvider(entries);
  context.subscriptions.push(
    vscode.window.registerTreeDataProvider(viewId, provider)
  );
  return provider;
}