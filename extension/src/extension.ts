import * as cp from "child_process";
import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";

export function activate(context: vscode.ExtensionContext) {
  const backend = new BackendClient();
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
      const pythonExtension = vscode.extensions.getExtension("ms-python.python");
      if (!pythonExtension) {
        vscode.window.showErrorMessage("VMECdash requires the Microsoft Python extension to select an interpreter.");
        return;
      }
      const override = getPythonOverride();
      await pythonExtension.activate();
      await vscode.commands.executeCommand("python.setInterpreter");
      backend.restart();
      if (override) {
        vscode.window.showWarningMessage(
          `VMECdash will still use ${override.source} (${override.python}) until that override is cleared.`,
        );
      }
      try {
        const resolution = await resolvePython();
        const health = await backend.request("health", {});
        vscode.window.showInformationMessage(
          `VMECdash backend OK using ${resolution.python} from ${resolution.source}: vmecdash ${health.vmecdashVersion}, jax ${health.jaxVersion}`,
        );
      } catch (error) {
        vscode.window.showErrorMessage(`VMECdash backend failed: ${messageOf(error)}`);
      }
    }),
    backend,
  );
}

export function deactivate() {}

type Pending = {
  resolve: (value: any) => void;
  reject: (reason?: any) => void;
  timer?: NodeJS.Timeout;
};

/** An Error that carries the backend's error code, so callers can branch on it. */
class BackendError extends Error {
  constructor(
    message: string,
    readonly code?: string,
  ) {
    super(message);
    this.name = "BackendError";
  }
}

// Field-line renders at the highest resolution take a couple of seconds; allow generous
// headroom, but never leave a request pending forever if the backend wedges.
const REQUEST_TIMEOUT_MS = 120000;

class BackendClient implements vscode.Disposable {
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private buffer = "";
  private stderr = "";
  private process?: cp.ChildProcessWithoutNullStreams;

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
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new BackendError(`VMECdash backend did not answer "${method}" within ${REQUEST_TIMEOUT_MS / 1000}s.`, "TIMEOUT"));
      }, REQUEST_TIMEOUT_MS);
      this.pending.set(id, { resolve, reject, timer });
      child.stdin.write(payload, (error) => {
        if (error) {
          this.settle(id);
          reject(error);
        }
      });
    });
  }

  /** Remove a pending entry and clear its timeout. Returns it if it was still live. */
  private settle(id: number): Pending | undefined {
    const pending = this.pending.get(id);
    if (!pending) return undefined;
    if (pending.timer) clearTimeout(pending.timer);
    this.pending.delete(id);
    return pending;
  }

  private rejectAll(error: Error) {
    for (const id of [...this.pending.keys()]) {
      this.settle(id)?.reject(error);
    }
  }

  private async ensureProcess(): Promise<cp.ChildProcessWithoutNullStreams> {
    if (this.process && !this.process.killed) {
      return this.process;
    }
    const { python, source } = await resolvePython();
    // Validate explicit interpreter paths up front (bare commands like "python3" resolve via PATH).
    const looksLikePath = python.includes("/") || python.includes("\\");
    if (looksLikePath && !fs.existsSync(python)) {
      throw new Error(`Python interpreter not found: "${python}" (from ${source}).`);
    }
    const config = vscode.workspace.getConfiguration("vmecdash");
    const extraArgs = config.get<string[]>("backendArgs") || [];
    const args = ["-m", "vmecdash.vscode_backend", "--stdio", ...extraArgs];
    const child = cp.spawn(python, args, {
      cwd: workspaceCwd(),
      // Force the backend's JAX onto the CPU so it never allocates GPU VRAM (important on
      // shared GPU clusters). An explicit JAX_PLATFORMS in the environment still wins.
      env: { ...process.env, JAX_PLATFORMS: process.env.JAX_PLATFORMS ?? "cpu", PYTHONUNBUFFERED: "1" },
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.process = child;
    this.stderr = "";
    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk) => this.onStdout(String(chunk)));
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk) => {
      const text = String(chunk);
      this.stderr = (this.stderr + text).slice(-4000);
      console.error(`[vmecdash-backend] ${text}`);
    });
    child.on("error", (err) => {
      this.rejectAll(new Error(`Failed to start Python "${python}" (from ${source}): ${messageOf(err)}`));
      this.process = undefined;
    });
    child.on("exit", (code, signal) => {
      this.rejectAll(this.describeExit(code, signal, python, source));
      this.process = undefined;
    });
    return child;
  }

  private describeExit(code: number | null, signal: NodeJS.Signals | null, python: string, source: string): Error {
    const stderr = this.stderr.trim();
    const importFailure = /No module named ['"]?vmecdash/.test(stderr) || /ModuleNotFoundError.*vmecdash/.test(stderr);
    if (importFailure) {
      showBackendSetupError(python, source);
      return new Error(
        `The Python interpreter (${python}, from ${source}) does not have the 'vmecdash' package. ` +
          `Run 'pip install vmecdash' into it, or set vmecdash.pythonPath to an interpreter that has it.`,
      );
    }
    const tail = stderr ? ` — ${stderr.split("\n").slice(-3).join(" ").slice(-300)}` : "";
    return new Error(`VMECdash backend exited (${code ?? signal})${tail}`);
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
      const pending = this.settle(response.id);
      if (!pending) continue;
      if (response.error) {
        // Keep the code alongside the message: the editor branches on SESSION_NOT_FOUND
        // to recover transparently when the backend has been restarted.
        pending.reject(new BackendError(response.error.message || response.error.code, response.error.code));
      } else {
        pending.resolve(response.result);
      }
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

    const openDocument = async () => {
      const meta = await this.backend.request("open", { path: document.uri.fsPath });
      sessionId = meta.sessionId;
      return meta;
    };
    const renderOnce = (id: string | undefined, message: any) =>
      this.backend.request("render", {
        sessionId: id,
        view: message.view,
        controls: message.controls,
        theme: message.theme,
      });
    const isMissingSession = (error: unknown) => error instanceof BackendError && error.code === "SESSION_NOT_FOUND";

    // Tell the panel when the file changes underneath it. Auto-reload is opt-in: a
    // running VMEC rewrites its wout repeatedly, and reloading mid-inspection would
    // yank the view out from under whoever is reading it.
    const watcher = vscode.workspace.createFileSystemWatcher(
      new vscode.RelativePattern(vscode.Uri.file(path.dirname(document.uri.fsPath)), path.basename(document.uri.fsPath)),
    );
    const announceChange = () => {
      const auto = vscode.workspace.getConfiguration("vmecdash").get<boolean>("autoReload") === true;
      webview.postMessage({ type: "stale", auto });
    };
    watcher.onDidChange(announceChange);
    watcher.onDidCreate(announceChange);

    webview.onDidReceiveMessage(async (message) => {
      try {
        if (message.type === "ready") {
          webview.postMessage({ type: "status", message: "Starting backend..." });
          const health = await this.backend.request("health", {});
          warnIfPlotlyUntested(health);
          webview.postMessage({ type: "opened", meta: await openDocument() });
        } else if (message.type === "refresh") {
          let meta: any;
          try {
            meta = await this.backend.request("refresh", { sessionId: sessionId ?? message.sessionId });
          } catch (error) {
            if (!isMissingSession(error)) throw error;
            meta = { ...(await openDocument()), changed: true };
          }
          if (meta.changed) {
            sessionId = meta.sessionId;
            webview.postMessage({ type: "opened", meta });
          } else {
            webview.postMessage({ type: "fresh" });
          }
        } else if (message.type === "render") {
          let result: any;
          try {
            result = await renderOnce(message.sessionId, message);
          } catch (error) {
            if (!isMissingSession(error)) throw error;
            // The backend was restarted (Select Python Interpreter does this) or the
            // session was retired. Re-open and retry once so the panel never dies.
            const meta = await openDocument();
            if (meta.sessionId !== message.sessionId) {
              // The file also changed while we were away - hand back fresh metadata and
              // let the webview re-render against the new schema.
              webview.postMessage({ type: "opened", meta });
              return;
            }
            result = await renderOnce(meta.sessionId, message);
          }
          webview.postMessage({ type: "rendered", serial: message.serial, result });
        } else if (message.type === "exportFigure") {
          // Plotly's own download writes a blob from inside the page, which a VS Code
          // webview blocks. Ask for the format here, have the webview rasterise, then
          // save through the extension host - the same route exportReport already uses.
          const picked = await vscode.window.showQuickPick(
            [
              { label: "SVG", description: "Vector - scales cleanly into a paper", format: "svg" },
              { label: "PNG", description: "Bitmap at 2x scale", format: "png" },
            ],
            { placeHolder: "Export the current figure as" },
          );
          if (picked) {
            webview.postMessage({ type: "renderImage", format: picked.format, scale: picked.format === "png" ? 2 : 1 });
          }
        } else if (message.type === "figureImage") {
          const comma = String(message.dataUrl || "").indexOf(",");
          if (comma < 0) throw new Error("The figure could not be rendered for export.");
          const payload = String(message.dataUrl).slice(comma + 1);
          // Plotly hands back SVG percent-encoded and raster formats base64-encoded.
          const bytes =
            message.format === "svg"
              ? Buffer.from(decodeURIComponent(payload), "utf8")
              : Buffer.from(payload, "base64");
          const root = vscode.workspace.workspaceFolders?.[0]?.uri || document.uri;
          const stem = path.basename(document.uri.fsPath).replace(/\.nc$/i, "");
          const target = await vscode.window.showSaveDialog({
            defaultUri: vscode.Uri.joinPath(root, `${stem}_${message.view || "figure"}.${message.format}`),
          });
          if (target) {
            await vscode.workspace.fs.writeFile(target, bytes);
            webview.postMessage({ type: "status", message: `Saved ${path.basename(target.fsPath)}` });
          }
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
      watcher.dispose();
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
      "{{brandUri}}": webview.asWebviewUri(vscode.Uri.joinPath(media, "brand.png")).toString(),
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

type PythonResolution = { python: string; source: string };

async function resolvePython(): Promise<PythonResolution> {
  const override = getPythonOverride();
  if (override) {
    return override;
  }
  const pythonExtension = vscode.extensions.getExtension("ms-python.python");
  if (pythonExtension) {
    try {
      const api = pythonExtension.isActive ? pythonExtension.exports : await pythonExtension.activate();
      const environmentPath = api?.environments?.getActiveEnvironmentPath
        ? api.environments.getActiveEnvironmentPath(pythonResource())
        : undefined;
      if (environmentPath?.path) {
        return { python: environmentPath.path, source: "the Python extension's selected interpreter" };
      }
      const details = api?.settings?.getExecutionDetails ? api.settings.getExecutionDetails() : undefined;
      if (details?.execCommand?.[0]) {
        return { python: details.execCommand[0], source: "the Python extension's selected interpreter" };
      }
    } catch (error) {
      console.warn(`Unable to read Python extension interpreter: ${messageOf(error)}`);
    }
  }
  const fallback = process.platform === "win32" ? "python" : "python3";
  return { python: fallback, source: `"${fallback}" on PATH` };
}

function getPythonOverride(): PythonResolution | undefined {
  const configPath = vscode.workspace.getConfiguration("vmecdash").get<string>("pythonPath");
  if (configPath && configPath.trim()) {
    return { python: configPath.trim(), source: "the vmecdash.pythonPath setting" };
  }
  const envPath = process.env.VMECDASH_PYTHON;
  if (envPath && envPath.trim()) {
    return { python: envPath.trim(), source: "the VMECDASH_PYTHON environment variable" };
  }
  return undefined;
}

function pythonResource(): vscode.Uri | undefined {
  return vscode.workspace.workspaceFolders?.[0]?.uri;
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

function showBackendSetupError(python: string, source: string) {
  const selectPython = "Select Python Interpreter";
  const openDocs = "Open Setup Docs";
  vscode.window
    .showErrorMessage(
      `VMECdash backend could not start: the Python interpreter (${python}, from ${source}) does not have the 'vmecdash' package installed. ` +
        `Install it with 'pip install vmecdash', or point the extension at an interpreter that has it.`,
      selectPython,
      openDocs,
    )
    .then((choice) => {
      if (choice === selectPython) {
        vscode.commands.executeCommand("vmecdash.selectPython");
      } else if (choice === openDocs) {
        vscode.env.openExternal(vscode.Uri.parse("https://github.com/DMCXE/VMECdash#readme"));
      }
    });
}
