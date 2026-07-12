(function () {
  const vscode = acquireVsCodeApi();

  // Inline SVG icons (theme via currentColor). Keys match ViewSpec.icon in view_schema.py.
  const ICONS = {
    overview:
      '<svg viewBox="0 0 16 16" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.3"><rect x="1.5" y="1.5" width="5" height="5" rx="1"/><rect x="9.5" y="1.5" width="5" height="5" rx="1"/><rect x="1.5" y="9.5" width="5" height="5" rx="1"/><rect x="9.5" y="9.5" width="5" height="5" rx="1"/></svg>',
    profiles:
      '<svg viewBox="0 0 16 16" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"><path d="M1.8 14.2V1.8"/><path d="M1.8 14.2h12.4"/><path d="M2.5 11.5 6 7.5 9 9.5 14 3.2"/></svg>',
    twod:
      '<svg viewBox="0 0 16 16" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.3"><ellipse cx="8" cy="8" rx="6.3" ry="4.3"/><ellipse cx="8" cy="8" rx="3.3" ry="2.1"/></svg>',
    threed:
      '<svg viewBox="0 0 16 16" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linejoin="round"><path d="M8 1.7 14.2 5v6L8 14.3 1.8 11V5Z"/><path d="M1.8 5 8 8.3 14.2 5"/><path d="M8 8.3v6"/></svg>',
    fieldline:
      '<svg viewBox="0 0 16 16" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"><path d="M1.5 8c2.2-4.6 4.3-4.6 6.5 0s4.3 4.6 6.5 0"/></svg>',
  };

  const state = {
    sessionId: null,
    meta: null,
    schema: null,
    currentView: "overview",
    renderSerial: 0,
    controls: {},
  };

  const el = {
    nav: document.getElementById("nav"),
    status: document.getElementById("status"),
    stats: document.getElementById("stats"),
    plot: document.getElementById("plot"),
    controls: document.getElementById("controlBody"),
    viewTitle: document.getElementById("viewTitle"),
    loading: document.getElementById("loading"),
  };

  const PLOT_CONFIG = { displaylogo: false, responsive: true, scrollZoom: true };
  let plotReady = false;

  // Plotly's responsive:true only reacts to WINDOW resizes. Observe the plot container so the
  // figure re-fits on the initial layout settle and on view switches (which change the stats
  // height and thus the plot area) without the user having to resize the VS Code window.
  const resizePlot = debounce(function () {
    if (!plotReady) return;
    try {
      Plotly.Plots.resize(el.plot);
    } catch (e) {
      /* plot not initialised yet */
    }
  }, 80);
  if (window.ResizeObserver) {
    new ResizeObserver(resizePlot).observe(el.plot);
  }

  function post(type, payload) {
    vscode.postMessage(Object.assign({ type }, payload || {}));
  }

  function setStatus(text, kind) {
    el.status.textContent = text || "";
    el.status.dataset.kind = kind || "";
  }

  function showLoading(on) {
    el.loading.classList.toggle("hidden", !on);
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[ch]));
  }

  function views() {
    return (state.schema && state.schema.views) || [];
  }

  function currentSpec() {
    return views().find((view) => view.id === state.currentView) || views()[0];
  }

  function controlSpec(id) {
    const spec = currentSpec();
    return spec && spec.controls.find((control) => control.id === id);
  }

  // Seed control defaults from the schema (single source of truth in Python).
  function seedDefaults() {
    views().forEach((view) => {
      (view.controls || []).forEach((control) => {
        if (state.controls[control.id] === undefined) state.controls[control.id] = control.default;
      });
    });
  }

  // visibleWhen from the schema: {siblingCtrlId: valueOrList}; AND across keys, list = any-of.
  // Values are string-compared, matching how <option> selection is resolved.
  function isVisible(control) {
    const cond = control.visibleWhen;
    if (!cond) return true;
    return Object.keys(cond).every((id) => {
      const wanted = cond[id];
      const actual = String(state.controls[id]);
      if (Array.isArray(wanted)) return wanted.some((v) => String(v) === actual);
      return String(wanted) === actual;
    });
  }

  // Controls whose value drives another control's visibility need a panel re-render on change.
  function isVisibilityDriver(id) {
    const spec = currentSpec();
    return !!spec && spec.controls.some((control) => control.visibleWhen && control.visibleWhen[id] !== undefined);
  }

  function optionsFor(spec) {
    if (spec.options) return spec.options;
    const meta = state.meta || {};
    if (spec.optionsFrom === "fields") return meta.fields || [];
    if (spec.optionsFrom === "profiles") return [...(meta.profiles || []), ...(meta.computedProfiles || [])];
    return [];
  }

  function optionHtml(options, selected) {
    return (options || [])
      .map((opt) => {
        const isSel = String(opt.value) === String(selected) ? " selected" : "";
        return `<option value="${escapeHtml(opt.value)}"${isSel}>${escapeHtml(opt.label || opt.value)}</option>`;
      })
      .join("");
  }

  function formatValue(spec, value) {
    const num = Number(value);
    const text = spec.step != null && Number(spec.step) < 1 ? num.toFixed(2) : String(num);
    return spec.unit ? `${text} ${spec.unit}` : text;
  }

  function renderNav() {
    el.nav.innerHTML = views()
      .map(
        (view) =>
          `<button class="nav-item${state.currentView === view.id ? " active" : ""}" data-view="${view.id}" title="${escapeHtml(view.label)}"><span class="nav-icon">${ICONS[view.icon] || ""}</span><span class="nav-label">${escapeHtml(view.label)}</span></button>`,
      )
      .join("");
    el.nav.querySelectorAll("[data-view]").forEach((button) => {
      button.addEventListener("click", () => {
        if (state.currentView === button.dataset.view) return;
        state.currentView = button.dataset.view;
        renderNav();
        renderControls();
        requestRender();
      });
    });
  }

  function controlRow(label, body, extraClass) {
    return `<div class="control${extraClass ? " " + extraClass : ""}"><span class="control-label">${escapeHtml(label)}</span>${body}</div>`;
  }

  function renderControl(control) {
    const value = state.controls[control.id];
    if (control.kind === "select") {
      return controlRow(
        control.label,
        `<div class="select-wrap"><select data-id="${control.id}">${optionHtml(optionsFor(control), value)}</select></div>`,
      );
    }
    if (control.kind === "slider") {
      return controlRow(
        control.label,
        `<div class="slider-row"><input class="slider" type="range" data-id="${control.id}" min="${control.min}" max="${control.max}" step="${control.step}" value="${value}"><output class="slider-val" id="out-${control.id}">${formatValue(control, value)}</output></div>`,
        "control-slider",
      );
    }
    if (control.kind === "number") {
      const attrs = [
        control.min != null ? `min="${control.min}"` : "",
        control.max != null ? `max="${control.max}"` : "",
        control.step != null ? `step="${control.step}"` : "",
      ]
        .filter(Boolean)
        .join(" ");
      return controlRow(control.label, `<input class="number" type="number" data-id="${control.id}" ${attrs} value="${value}">`);
    }
    if (control.kind === "checkbox") {
      return `<label class="control control-toggle"><span class="control-label">${escapeHtml(control.label)}</span><span class="toggle"><input type="checkbox" data-id="${control.id}"${value ? " checked" : ""}><span class="toggle-track"></span></span></label>`;
    }
    return "";
  }

  function renderControls() {
    const spec = currentSpec();
    if (el.viewTitle) el.viewTitle.textContent = spec ? spec.label : "";
    if (!spec || !spec.controls.length) {
      el.controls.innerHTML = '<p class="control-empty">This view has no adjustable controls. It uses canonical VMEC profiles and scalar metadata.</p>';
      return;
    }
    el.controls.innerHTML = spec.controls.filter(isVisible).map(renderControl).join("");
    bindControls();
  }

  function bindControls() {
    el.controls.querySelectorAll("input,select").forEach((input) => {
      const isRange = input.type === "range";
      if (isRange) {
        // Live value readout on every drag frame (no render spam).
        input.addEventListener("input", () => {
          const out = document.getElementById("out-" + input.dataset.id);
          const spec = controlSpec(input.dataset.id);
          if (out && spec) out.textContent = formatValue(spec, input.value);
        });
      }
      const event = isRange ? "input" : "change";
      input.addEventListener(
        event,
        debounce(() => {
          readControls();
          // Selects/checkboxes may show/hide sibling controls (schema visibleWhen).
          if (input.dataset.id && isVisibilityDriver(input.dataset.id)) renderControls();
          // Slider drags render silently (no overlay flash); pumpRender paces the stream.
          requestRender({ silent: isRange });
        }, isRange ? 60 : 0),
      );
    });
  }

  function readControls() {
    el.controls.querySelectorAll("input,select").forEach((input) => {
      const id = input.dataset.id;
      if (!id) return;
      if (input.type === "checkbox") state.controls[id] = input.checked;
      else if (input.type === "number" || input.type === "range") state.controls[id] = Number(input.value);
      else state.controls[id] = input.value;
    });
  }

  // One render request in flight at a time; newer requests coalesce into `renderQueued` so a
  // dragged slider produces a stream of sequential renders paced by backend latency instead of
  // flooding the (serial) backend. `silent` skips the loading overlay for continuous updates.
  let renderInFlight = false;
  let renderQueued = null;

  function requestRender(opts) {
    if (!state.sessionId) return;
    renderQueued = { serial: ++state.renderSerial, silent: !!(opts && opts.silent) };
    pumpRender();
  }

  function pumpRender() {
    if (renderInFlight || !renderQueued) return;
    const req = renderQueued;
    renderQueued = null;
    renderInFlight = true;
    setStatus("Rendering…", "busy");
    if (!req.silent) showLoading(true);
    post("render", {
      serial: req.serial,
      sessionId: state.sessionId,
      view: state.currentView,
      controls: state.controls,
      theme: document.body.classList.contains("vscode-light") ? "light" : "dark",
    });
  }

  function statCard(item) {
    return `<div class="stat"><div class="stat-title">${escapeHtml(item.title)}</div><div class="stat-value">${escapeHtml(item.value)} ${escapeHtml(item.unit || "")}</div></div>`;
  }

  function renderStats(stats) {
    // Hero KPIs and detail cards go into separate responsive grids so short and tall cards do
    // not share a row (which made the dashboard look scrambled). Order is preserved.
    if (stats && stats.hero && stats.details) {
      const hero = stats.hero.map(statCard).join("");
      const details = stats.details
        .map((group) => {
          const rows = (group.items || [])
            .map((item) => `<div class="detail-row"><span class="label">${escapeHtml(item.label)}</span><span>${escapeHtml(item.value)} ${escapeHtml(item.unit || "")}</span></div>`)
            .join("");
          return `<div class="detail"><strong>${escapeHtml(group.title)}</strong>${rows}</div>`;
        })
        .join("");
      el.stats.innerHTML = `<div class="stat-grid">${hero}</div><div class="detail-grid">${details}</div>`;
      return;
    }
    const base = (stats && stats.base) || [];
    el.stats.innerHTML = `<div class="stat-grid">${base.map(statCard).join("")}</div>`;
  }

  function debounce(fn, ms) {
    let timer;
    return function () {
      clearTimeout(timer);
      timer = setTimeout(fn, ms);
    };
  }

  window.addEventListener("message", (event) => {
    const message = event.data;
    if (message.type === "opened") {
      state.meta = message.meta;
      state.schema = message.meta.schema;
      state.sessionId = message.meta.sessionId;
      const first = views()[0];
      if (first) state.currentView = first.id;
      seedDefaults();
      renderNav();
      renderControls();
      requestRender();
    } else if (message.type === "rendered") {
      renderInFlight = false;
      pumpRender();
      if (message.serial && message.serial < state.renderSerial) return; // drop stale response
      showLoading(false);
      const figure = message.result.figure;
      renderStats(message.result.stats);
      Plotly.react(el.plot, figure.data || [], figure.layout || {}, PLOT_CONFIG).then(function () {
        plotReady = true;
        try {
          Plotly.Plots.resize(el.plot);
        } catch (e) {
          /* ignore */
        }
      });
      setStatus("Ready", "ok");
    } else if (message.type === "error") {
      renderInFlight = false;
      pumpRender();
      showLoading(false);
      setStatus(message.message || "Error", "error");
    } else if (message.type === "status") {
      setStatus(message.message);
    }
  });

  const exportButton = document.getElementById("exportReport");
  if (exportButton) exportButton.addEventListener("click", () => post("exportReport", { sessionId: state.sessionId }));

  // ---- Layout states: auto-collapse from pane width (matchMedia), plus persisted manual
  // toggles that work at any width. navCollapsed stays undefined (= follow width) until the
  // user explicitly toggles the sidebar.
  const shell = document.querySelector(".shell");
  const navToggle = document.getElementById("navToggle");
  const controlsToggle = document.getElementById("controlsToggle");
  const railAuto = window.matchMedia("(max-width: 1100px)");
  const drawerAuto = window.matchMedia("(max-width: 900px)");
  const ui = Object.assign({ navCollapsed: undefined, controlsHidden: false, drawerOpen: false }, vscode.getState());

  function saveUi() {
    vscode.setState(Object.assign({}, vscode.getState(), ui));
  }

  function applyLayout() {
    const rail = ui.navCollapsed !== undefined ? ui.navCollapsed : railAuto.matches;
    shell.classList.toggle("nav-rail", rail);
    shell.classList.toggle("controls-drawer", drawerAuto.matches);
    shell.classList.toggle("controls-open", drawerAuto.matches && ui.drawerOpen);
    shell.classList.toggle("controls-hidden", !drawerAuto.matches && ui.controlsHidden);
    if (navToggle) navToggle.setAttribute("aria-expanded", String(!rail));
    if (controlsToggle) {
      controlsToggle.setAttribute("aria-expanded", String(drawerAuto.matches ? ui.drawerOpen : !ui.controlsHidden));
    }
  }

  if (navToggle) {
    navToggle.addEventListener("click", () => {
      ui.navCollapsed = !shell.classList.contains("nav-rail");
      saveUi();
      applyLayout();
    });
  }
  if (controlsToggle) {
    controlsToggle.addEventListener("click", () => {
      if (drawerAuto.matches) ui.drawerOpen = !ui.drawerOpen;
      else ui.controlsHidden = !ui.controlsHidden;
      saveUi();
      applyLayout();
    });
  }
  document.querySelector(".plot-wrap").addEventListener("click", () => {
    if (drawerAuto.matches && ui.drawerOpen) {
      ui.drawerOpen = false;
      saveUi();
      applyLayout();
    }
  });
  railAuto.addEventListener("change", applyLayout);
  drawerAuto.addEventListener("change", applyLayout);
  applyLayout();

  setStatus("Starting backend…", "busy");
  post("ready");
})();
