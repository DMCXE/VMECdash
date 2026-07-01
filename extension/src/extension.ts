import * as cp from "child_process";
import * as fs from "fs";
import * as vscode from "vscode";

export function activate(context: vscode.ExtensionContext) {
  const backend = new BackendClient(context);
  const provider = new VmecDashEditorProvider(context, backend);
  context.subscriptions.push(
    vscode.window.registerCustomEditorProvider("vmecdash.woutPreview", provider, {
      webviewOptions: { retainContextWhenHidden: true },
      supportsMultipleEditorsPerDocument: false,
    }),
  );
  context.subscriptions.push(
    vscode.commands.registerCommand("vmecdash.openPreview", async (uri?: vscode.Uri) => {
      let target = uri;
      if (!target) {
        const picked = await vscode.window.showOpenDialog({
          canSelectMany: false,
          filters: { NetCDF: ["nc"], "All files": ["*"] },
        });
        target = picked?.[0];
      }
      if (target) {
        await vscode.commands.executeCommand("vscode.openWith", target, "vmecdash.woutPreview");
      }
    }),
    vscode.commands.registerCommand("vmecdash.checkBackend", async () => {
      try {
        const health = await backend.request("health", {});
        vscode.window.showInformationMessage(
          `VMECdash backend OK: vmecdash ${health.vmecdashVersion}, jax ${health.jaxVersion}, plotly.py ${health.plotlyPythonVersion}`,
        );
      } catch (error) {
        vscode.window.showErrorMessage(`VMECdash backend failed: ${messageOf(error)}`);
      }
    }),
    vscode.commands.registerCommand("vmecdash.selectPython", async () => {
      const current = vscode.workspace.getConfiguration("vmecdash").get<string>("pythonPath") || "";
      const value = await vscode.window.showInputBox({
        title: "VMECdash Python Path",
        prompt: "Python executable with vmecdash installed",
        value: current,
      });
      if (value !== undefined) {
        await vscode.workspace.getConfiguration("vmecdash").update("pythonPath", value, vscode.ConfigurationTarget.Workspace);
        backend.restart();
      }
    }),
    backend,
  );
}

export function deactivate() {}

type Pending = {
  resolve: (value: any) => void;
  reject: (reason?: any) => void;
};

class BackendClient implements vscode.Disposable {
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private buffer = "";
  private process?: cp.ChildProcessWithoutNullStreams;

  constructor(private readonly context: vscode.ExtensionContext) {}

  dispose() {
    if (this.process) {
      this.process.kill();
      this.process = undefined;
    }
  }

  restart() {
    this.dispose();
  }

  async request(method: string, params: any): Promise<any> {
    const child = await this.ensureProcess();
    const id = this.nextId++;
    const payload = JSON.stringify({ id, method, params: params || {} }) + "\n";
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      child.stdin.write(payload, (error) => {
        if (error) {
          this.pending.delete(id);
          reject(error);
        }
      });
    });
  }

  private async ensureProcess(): Promise<cp.ChildProcessWithoutNullStreams> {
    if (this.process && !this.process.killed) {
      return this.process;
    }
    const python = await resolvePython();
    const config = vscode.workspace.getConfiguration("vmecdash");
    const extraArgs = config.get<string[]>("backendArgs") || [];
    const args = ["-m", "vmecdash.vscode_backend", "--stdio", ...extraArgs];
    const child = cp.spawn(python, args, {
      cwd: workspaceCwd(),
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.process = child;
    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk) => this.onStdout(String(chunk)));
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk) => console.error(`[vmecdash-backend] ${chunk}`));
    child.on("exit", (code, signal) => {
      const error = new Error(`VMECdash backend exited (${code ?? signal})`);
      for (const { reject } of this.pending.values()) reject(error);
      this.pending.clear();
      this.process = undefined;
    });
    return child;
  }

  private onStdout(chunk: string) {
    this.buffer += chunk;
    let idx: number;
    while ((idx = this.buffer.indexOf("\n")) >= 0) {
      const line = this.buffer.slice(0, idx).trim();
      this.buffer = this.buffer.slice(idx + 1);
      if (!line) continue;
      let response: any;
      try {
        response = JSON.parse(line);
      } catch {
        console.error(`Invalid VMECdash backend JSON: ${line}`);
        continue;
      }
      const pending = this.pending.get(response.id);
      if (!pending) continue;
      this.pending.delete(response.id);
      if (response.error) pending.reject(new Error(response.error.message || response.error.code));
      else pending.resolve(response.result);
    }
  }
}

class VmecDashEditorProvider implements vscode.CustomReadonlyEditorProvider<{ uri: vscode.Uri; dispose(): void }> {
  constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly backend: BackendClient,
  ) {}

  async openCustomDocument(uri: vscode.Uri) {
    return { uri, dispose() {} };
  }

  async resolveCustomEditor(document: { uri: vscode.Uri }, webviewPanel: vscode.WebviewPanel) {
    const webview = webviewPanel.webview;
    webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this.context.extensionUri, "media")],
    };
    webview.html = this.htmlFor(webview);
    let sessionId: string | undefined;
    webview.onDidReceiveMessage(async (message) => {
      try {
        if (message.type === "ready") {
          webview.postMessage({ type: "status", message: "Starting backend..." });
          const health = await this.backend.request("health", {});
          warnIfPlotlyUntested(health);
          const meta = await this.backend.request("open", { path: document.uri.fsPath });
          sessionId = meta.sessionId;
          webview.postMessage({ type: "opened", meta });
        } else if (message.type === "render") {
          const result = await this.backend.request("render", {
            sessionId: message.sessionId,
            view: message.view,
            controls: message.controls,
            theme: message.theme,
          });
          webview.postMessage({ type: "rendered", serial: message.serial, result });
        } else if (message.type === "exportReport") {
          const report = await this.backend.request("exportReport", { sessionId: message.sessionId });
          const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri || document.uri;
          const target = await vscode.window.showSaveDialog({ defaultUri: vscode.Uri.joinPath(workspaceRoot, report.filename) });
          if (target) {
            await vscode.workspace.fs.writeFile(target, Buffer.from(report.content, "utf8"));
          }
        }
      } catch (error) {
        webview.postMessage({ type: "error", message: messageOf(error) });
      }
    });
    webviewPanel.onDidDispose(() => {
      if (sessionId) this.backend.request("dispose", { sessionId }).catch(() => undefined);
    });
  }

  private htmlFor(webview: vscode.Webview): string {
    const media = vscode.Uri.joinPath(this.context.extensionUri, "media");
    const templatePath = vscode.Uri.joinPath(media, "index.html").fsPath;
    const nonce = String(Date.now()) + String(Math.random()).slice(2);
    const values: Record<string, string> = {
      "{{nonce}}": nonce,
      "{{cspSource}}": webview.cspSource,
      "{{stylesUri}}": webview.asWebviewUri(vscode.Uri.joinPath(media, "styles.css")).toString(),
      "{{plotlyUri}}": webview.asWebviewUri(vscode.Uri.joinPath(media, "plotly.min.js")).toString(),
      "{{scriptUri}}": webview.asWebviewUri(vscode.Uri.joinPath(media, "main.js")).toString(),
    };
    let html = fs.readFileSync(templatePath, "utf8");
    for (const [key, value] of Object.entries(values)) {
      html = html.split(key).join(value);
    }
    return html;
  }
}

async function resolvePython(): Promise<string> {
  const configPath = vscode.workspace.getConfiguration("vmecdash").get<string>("pythonPath");
  if (configPath) return configPath;
  const pythonExtension = vscode.extensions.getExtension("ms-python.python");
  if (pythonExtension) {
    try {
      const api = pythonExtension.isActive ? pythonExtension.exports : await pythonExtension.activate();
      const details = api?.settings?.getExecutionDetails ? api.settings.getExecutionDetails() : undefined;
      if (details?.execCommand?.[0]) return details.execCommand[0];
    } catch (error) {
      console.warn(`Unable to read Python extension interpreter: ${messageOf(error)}`);
    }
  }
  return process.platform === "win32" ? "python" : "python3";
}

function workspaceCwd(): string | undefined {
  return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
}

function warnIfPlotlyUntested(health: any) {
  const version = String(health?.plotlyPythonVersion || "");
  if (version && !version.startsWith("5.24.")) {
    vscode.window.showWarningMessage(
      `VMECdash was tested with plotly.py 5.24.x and bundled plotly.js 2.35.2; current plotly.py is ${version}.`,
    );
  }
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
